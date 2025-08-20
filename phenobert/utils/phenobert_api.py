# -*- coding: utf-8 -*-
"""
PhenoBERT API
Pre-loaded model wrapper for efficient HPO term tagging
"""

import os
import time
import logging
import warnings
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Import PhenoBERT modules
from api import annotate_text, get_most_related_HPO_term
from util import HPOTree, processStr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.simplefilter("ignore")

class PhenoBERTAPI:
    """
    Pre-loaded PhenoBERT API for efficient HPO term annotation
    
    This class provides a simple interface for text annotation using PhenoBERT.
    """
    
    def __init__(self):
        """
        Initialize PhenoBERT API with pre-loaded models
        """
        logger.info("Initializing PhenoBERT API...")
        
        try:
            # Initialize HPO tree (this will load all required models internally)
            self.hpo_tree = HPOTree()
            logger.info("PhenoBERT API initialized successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize PhenoBERT API: {e}")
            raise
    
    def annotate_text(self, text: str, 
                     param1: float = 0.8,
                     param2: float = 0.6, 
                     param3: float = 0.9,
                     use_longest: bool = True,
                     use_step_3: bool = True) -> Dict[str, str]:
        """
        Annotate text with HPO terms using PhenoBERT
        
        Args:
            text: Input text to annotate
            param1: Model parameter 1 (Layer 1 threshold)
            param2: Model parameter 2 (Sub-layer threshold)
            param3: Model parameter 3 (BERT matching threshold)
            use_longest: Whether to return only longest concepts
            use_step_3: Whether to use BERT matching step
            
        Returns:
            Dictionary with text as key and annotation info as value
            Format: {
                text: "hpo_term1 (HPO:xxxxx);hpo_term2 (HPO:xxxxx);...",
                "hpo_ids": "HPO:xxxxx;HPO:xxxxx;...",
                "raw_results": "raw annotation output"
            }
        """
        if not text or not text.strip():
            return {
                text: "",
                "hpo_ids": "",
                "raw_results": ""
            }
        
        try:
            # Call PhenoBERT annotation
            raw_result = annotate_text(
                text=text,
                output=None,  # Return string instead of writing to file
                param1=param1,
                param2=param2,
                param3=param3,
                use_longest=use_longest,
                use_step_3=use_step_3
            )
            
            # Parse results to extract HPO terms and IDs
            hpo_terms, hpo_ids = self._parse_results(raw_result)
            
            return {
                text: hpo_terms,
                "hpo_ids": hpo_ids,
                "raw_results": raw_result
            }
            
        except Exception as e:
            logger.error(f"Error during annotation: {e}")
            return {
                text: "",
                "hpo_ids": "",
                "raw_results": ""
            }
    
    def _parse_results(self, raw_result: str) -> Tuple[str, str]:
        """
        Parse raw annotation results into formatted strings
        
        Args:
            raw_result: Raw output from PhenoBERT annotation
            
        Returns:
            Tuple of (formatted_terms, hpo_ids_only)
        """
        if not raw_result:
            return "", ""
        
        formatted_terms = []
        hpo_ids = []
        
        # Parse each line of results
        lines = raw_result.strip().split('\n')
        for line in lines:
            if line.strip():
                parts = line.split('\t')
                if len(parts) >= 4:
                    # Extract: start, end, phrase, hpo_id, score, [neg_flag]
                    phrase = parts[2]
                    hpo_id = parts[3]
                    
                    # Get HPO term name from ID
                    hpo_term = self._get_hpo_term_name(hpo_id)
                    
                    # Format as "term (HPO:xxxxx)"
                    if hpo_term:
                        formatted_term = f"{hpo_term} ({hpo_id})"
                    else:
                        formatted_term = f"({hpo_id})"
                    
                    formatted_terms.append(formatted_term)
                    hpo_ids.append(hpo_id)
        
        return ";".join(formatted_terms), ";".join(hpo_ids)
    
    def _get_hpo_term_name(self, hpo_id: str) -> str:
        """
        Get HPO term name from HPO ID
        
        Args:
            hpo_id: HPO identifier (e.g., "HP:0000001")
            
        Returns:
            HPO term name or empty string if not found
        """
        try:
            if hpo_id in self.hpo_tree.data:
                name_list = self.hpo_tree.data[hpo_id].get('Name', [])
                if name_list and len(name_list) > 0:
                    return name_list[0]
            return ""
        except Exception:
            return ""
    
    def get_related_hpo_terms(self, phrases: List[str],
                             param1: float = 0.8,
                             param2: float = 0.6,
                             param3: float = 0.9) -> List[Tuple[str, str]]:
        """
        Get most related HPO terms for given phrases
        
        Args:
            phrases: List of phrases to analyze
            param1: Model parameter 1
            param2: Model parameter 2  
            param3: Model parameter 3
            
        Returns:
            List of (phrase, hpo_id) tuples
        """
        try:
            # Process phrases to remove numbers and normalize
            processed_phrases = []
            for phrase in phrases:
                processed = processStr(phrase)
                if processed:  # Only add non-empty processed phrases
                    processed_phrases.append(" ".join(processed))
            
            if not processed_phrases:
                return []
            
            # Get related HPO terms
            results = get_most_related_HPO_term(
                processed_phrases, 
                param1=param1, 
                param2=param2, 
                param3=param3
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting related HPO terms: {e}")
            return []
    
    def annotate_batch(self, texts: List[str], **kwargs) -> Dict[str, Dict[str, str]]:
        """
        Annotate multiple texts using parallel processing with 10 threads
        並行處理版本 - 注意：只有在確認模型線程安全時才使用
        
        Args:
            texts: List of texts to annotate
            **kwargs: Parameters passed to annotate_text
            
        Returns:
            Dictionary mapping each text to its annotation info
        """
        if not texts:
            return {}
        
        # 固定使用 10 個執行緒
        max_workers = 10
        show_progress = kwargs.pop('show_progress', True)
        
        # 預分配結果數組
        results = [None] * len(texts)
        
        def process_single(args):
            idx, text = args
            try:
                result = self.annotate_text(text, **kwargs)
                annotation_result = {
                    "hpo_terms": result[text],
                    "hpo_ids": result["hpo_ids"]
                }
                return idx, text, annotation_result
            except Exception as e:
                logger.error(f"Error processing index {idx}: {e}")
                return idx, text, {
                    "hpo_terms": "",
                    "hpo_ids": ""
                }
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single, (i, text)) 
                      for i, text in enumerate(texts)]
            
            iterator = tqdm(futures, desc="Processing in parallel") if show_progress else futures
            
            for future in iterator:
                idx, text, annotation_result = future.result()
                results[idx] = (text, annotation_result)
        
        # 轉換為原始格式的字典
        final_results = {}
        for text, annotation_result in results:
            final_results[text] = annotation_result
        
        return final_results
    
    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about loaded model and configuration
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': 'PhenoBERT',
            'api_version': '1.0',
            'description': 'PhenoBERT HPO term annotation system',
            'hpo_concepts': str(len(self.hpo_tree.hpo_list)) if hasattr(self.hpo_tree, 'hpo_list') else 'unknown'
        }


def create_api(**kwargs) -> PhenoBERTAPI:
    """
    Convenience function to create PhenoBERT API instance
    
    Returns:
        Initialized PhenoBERTAPI instance
    """
    return PhenoBERTAPI()


# Example usage and testing
if __name__ == "__main__":
    # Test the API
    print("Testing PhenoBERT API...")
    
    try:
        # Initialize API
        api = create_api()
        
        # Test text
        test_text = "The patient presented with seizures, intellectual disability, and microcephaly."
        
        # Annotate text
        print(f"\nInput text: {test_text}")
        result = api.annotate_text(test_text)
        print(f"HPO annotations: {result[test_text]}")
        print(f"HPO IDs: {result['hpo_ids']}")
        
        # Test batch annotation (always parallel with 10 threads)
        test_texts = [
            "Patient has fever and headache.",
            "Observed growth retardation and developmental delay.",
            "The patient shows signs of muscular dystrophy.",
            "Cardiac abnormalities were detected during examination.",
            "Patient exhibits autistic behavior and social communication deficits."
        ]
        
        print(f"\n=== Testing Parallel Processing (10 threads, {len(test_texts)} texts) ===")
        start_time = time.time()
        batch_results = api.annotate_batch(test_texts)
        processing_time = time.time() - start_time
        print(f"Parallel processing time: {processing_time:.2f} seconds")
        
        print("\nBatch results:")
        for text, annotation in batch_results.items():
            print(f"Text: {text}")
            print(f"HPO: {annotation['hpo_terms']}")
            print(f"IDs: {annotation['hpo_ids']}\n")
            
        # Model info
        info = api.get_model_info()
        print("Model info:", info)
        
        # Test phrase matching
        test_phrases = ["seizures", "intellectual disability"]
        related_terms = api.get_related_hpo_terms(test_phrases)
        print(f"\nRelated HPO terms for {test_phrases}:")
        for phrase, hpo_id in related_terms:
            print(f"  {phrase} -> {hpo_id}")
        
    except Exception as e:
        print(f"Error testing API: {e}")
        import traceback
        traceback.print_exc()