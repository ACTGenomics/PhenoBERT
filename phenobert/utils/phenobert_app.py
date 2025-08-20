# -*- coding: utf-8 -*-
"""
PhenoBERT FastAPI Application
Simple REST API for HPO term annotation using PhenoBERT
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
import time

# Import our PhenoBERT API
from phenobert_api import PhenoBERTAPI, create_api

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global API instance
phenobert_api = None

# Create FastAPI app
app = FastAPI(
    title="PhenoBERT API",
    description="Human Phenotype Ontology (HPO) term annotation service using PhenoBERT",
    version="1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize API on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the PhenoBERT API"""
    global phenobert_api
    
    logger.info("Starting PhenoBERT API...")
    try:
        phenobert_api = create_api()
        logger.info("PhenoBERT API initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize PhenoBERT API: {}".format(e))
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup the PhenoBERT API"""
    global phenobert_api
    
    logger.info("Shutting down PhenoBERT API...")
    phenobert_api = None

# Pydantic models for request/response
class AnnotateRequest(BaseModel):
    text: str = Field(..., description="Text to annotate with HPO terms", min_length=1)
    param1: Optional[float] = Field(0.8, description="Model parameter 1 (Layer 1 threshold)", ge=0.0, le=1.0)
    param2: Optional[float] = Field(0.6, description="Model parameter 2 (Sub-layer threshold)", ge=0.0, le=1.0)
    param3: Optional[float] = Field(0.9, description="Model parameter 3 (BERT matching threshold)", ge=0.0, le=1.0)
    use_longest: Optional[bool] = Field(True, description="Return only longest overlapping concepts")
    use_step_3: Optional[bool] = Field(True, description="Use BERT matching step")

class BatchAnnotateRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to annotate", min_items=1)
    param1: Optional[float] = Field(0.8, description="Model parameter 1", ge=0.0, le=1.0)
    param2: Optional[float] = Field(0.6, description="Model parameter 2", ge=0.0, le=1.0)
    param3: Optional[float] = Field(0.9, description="Model parameter 3", ge=0.0, le=1.0)
    use_longest: Optional[bool] = Field(True, description="Return only longest overlapping concepts")
    use_step_3: Optional[bool] = Field(True, description="Use BERT matching step")

class RelatedTermsRequest(BaseModel):
    phrases: List[str] = Field(..., description="List of phrases to find related HPO terms", min_items=1)
    param1: Optional[float] = Field(0.8, description="Model parameter 1", ge=0.0, le=1.0)
    param2: Optional[float] = Field(0.6, description="Model parameter 2", ge=0.0, le=1.0)
    param3: Optional[float] = Field(0.9, description="Model parameter 3", ge=0.0, le=1.0)

class AnnotateResponse(BaseModel):
    text: str
    hpo_terms: str
    hpo_ids: str = Field(..., description="HPO IDs only: HP:xxxxx;HP:xxxxx")
    processing_time: float

class BatchAnnotateResponse(BaseModel):
    results: List[AnnotateResponse]
    total_processing_time: float
    count: int

class RelatedTermsResponse(BaseModel):
    results: List[Dict[str, str]]
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    model_info: Dict
    timestamp: float

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# API Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic information"""
    return {
        "service": "PhenoBERT API",
        "version": "1.0",
        "description": "HPO term annotation service using PhenoBERT",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if phenobert_api is None:
        raise HTTPException(status_code=503, detail="PhenoBERT API not initialized")
    
    try:
        model_info = phenobert_api.get_model_info()
        return HealthResponse(
            status="healthy",
            model_info=model_info,
            timestamp=time.time()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/annotate", response_model=AnnotateResponse)
async def annotate_text_endpoint(request: AnnotateRequest):
    """
    Annotate a single text with HPO terms using PhenoBERT
    
    Returns HPO terms in format: "term1 (HP:xxxxx);term2 (HP:xxxxx);..."
    """
    if phenobert_api is None:
        raise HTTPException(status_code=503, detail="PhenoBERT API not initialized")
    
    try:
        start_time = time.time()
        
        # Call annotation API
        result = phenobert_api.annotate_text(
            text=request.text,
            param1=request.param1,
            param2=request.param2,
            param3=request.param3,
            use_longest=request.use_longest,
            use_step_3=request.use_step_3
        )
        
        processing_time = time.time() - start_time
        
        return AnnotateResponse(
            text=request.text,
            hpo_terms=result[request.text],
            hpo_ids=result["hpo_ids"],
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Annotation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Annotation failed: {str(e)}")

@app.post("/annotate/batch", response_model=BatchAnnotateResponse)
async def annotate_batch_endpoint(request: BatchAnnotateRequest):
    """
    Annotate multiple texts with HPO terms using PhenoBERT
    """
    if phenobert_api is None:
        raise HTTPException(status_code=503, detail="PhenoBERT API not initialized")
    
    # Optional: Add warning for very large batches
    if len(request.texts) > 100:
        logger.warning(f"Processing large batch of {len(request.texts)} texts. This may take a while.")
    
    try:
        start_time = time.time()
        
        # Call batch annotation API
        batch_results = phenobert_api.annotate_batch(
            texts=request.texts,
            param1=request.param1,
            param2=request.param2,
            param3=request.param3,
            use_longest=request.use_longest,
            use_step_3=request.use_step_3
        )
        
        total_processing_time = time.time() - start_time
        
        # Format results
        results = []
        for text in request.texts:
            annotation_info = batch_results.get(text, {"hpo_terms": "", "hpo_ids": ""})
            results.append(AnnotateResponse(
                text=text,
                hpo_terms=annotation_info["hpo_terms"],
                hpo_ids=annotation_info["hpo_ids"],
                processing_time=total_processing_time / len(request.texts)  # Average time
            ))
        
        return BatchAnnotateResponse(
            results=results,
            total_processing_time=total_processing_time,
            count=len(request.texts)
        )
        
    except Exception as e:
        logger.error(f"Batch annotation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch annotation failed: {str(e)}")

@app.post("/related_terms", response_model=RelatedTermsResponse)
async def get_related_terms(request: RelatedTermsRequest):
    """
    Get most related HPO terms for given phrases
    """
    if phenobert_api is None:
        raise HTTPException(status_code=503, detail="PhenoBERT API not initialized")
    
    try:
        start_time = time.time()
        
        # Get related terms
        related_terms = phenobert_api.get_related_hpo_terms(
            phrases=request.phrases,
            param1=request.param1,
            param2=request.param2,
            param3=request.param3
        )
        
        processing_time = time.time() - start_time
        
        # Format results
        results = []
        for phrase, hpo_id in related_terms:
            results.append({
                "phrase": phrase,
                "hpo_id": hpo_id
            })
        
        return RelatedTermsResponse(
            results=results,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Related terms error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Getting related terms failed: {str(e)}")

@app.get("/config")
async def get_config():
    """Get current API configuration"""
    if phenobert_api is None:
        raise HTTPException(status_code=503, detail="PhenoBERT API not initialized")
    
    try:
        model_info = phenobert_api.get_model_info()
        
        return {
            "model_info": model_info,
            "supported_parameters": {
                "param1": "Layer 1 threshold (0.0-1.0)",
                "param2": "Sub-layer threshold (0.0-1.0)", 
                "param3": "BERT matching threshold (0.0-1.0)",
                "use_longest": "Return only longest concepts",
                "use_step_3": "Use BERT matching step"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.detail).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )

# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "phenobert_app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )