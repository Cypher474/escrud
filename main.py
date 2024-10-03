from fastapi import FastAPI, HTTPException, UploadFile, File,Form,Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import io
import os

load_dotenv()

# Initialize the FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"])
# Fetch environment variables for OpenAI
api_key = os.getenv("OPENAI_API_KEY")

# Check if API key exists
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables")

# Initialize OpenAI client with the API key
client = openai.OpenAI(api_key=api_key)

# Data model for creating a new vector store
class VectorStoreCreate(BaseModel):
    name: str

# POST: Delete a vector store by ID
class VectorStoreDelete(BaseModel):
    vector_store_id: str

class UploadRequest(BaseModel):
    vector_store_id: str

class FileDeleteRequest(BaseModel):
    file_id: str
class VectorStoreRequest(BaseModel):
    vector_store_id: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/assistants/")
async def list_assistants():
    try:
        assistant_data = client.beta.assistants.list()
        assistants = []

        # Loop through the data to extract necessary fields
        for assistant in assistant_data.data:
            assistant_id = assistant.id
            assistant_name = assistant.name if assistant.name else 'No Name'
            assistant_model = assistant.model if hasattr(assistant, 'model') else 'Unknown Model'

            # Append to the list
            assistants.append({
                "name": assistant_name,
                "id": assistant_id,
                "model": assistant_model
            })

        return assistants
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error retrieving assistants: {e}")


@app.get("/files/")
async def list_files():
    try:
        file_list = client.files.list()
        files = []
        for file in file_list.data:
            file_id = file.id
            filename = file.filename
            files.append({"id": file_id, "filename": filename})
        return files
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error retrieving files: {e}")
    
# GET: List all vector stores
@app.get("/vector-stores/")
async def list_vector_stores():
    try:
        vector_list = client.beta.vector_stores.list()
        stores = []
        for vector_store in vector_list.data:
            vector_id = vector_store.id
            file_count = vector_store.file_counts.total
            name = vector_store.name if vector_store.name else 'No Name'
            stores.append({"name": name, "id": vector_id, "file_count": file_count})
        return stores
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error retrieving vector stores: {e}")
    
@app.delete("/files/delete")
async def delete_file(file_id: str = Query(..., description="The ID of the file to be deleted")):
    try:
        # Call the client method to delete the file
        response = client.files.delete(file_id)
        
        # which indicates success if it doesn't raise an exception
        return {"message": f"File {file_id} deleted successfully"}
    except Exception as e:
        # Handle specific exceptions if needed
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vector-store/files/")
async def list_vector_store_files(request: VectorStoreRequest):
    try:
        # Retrieve the vector store ID from the request
        vector_store_id = request.vector_store_id

        # Call OpenAI API to list files in the vector store by the vector_store_id
        vector_store_files = client.beta.vector_stores.files.list(vector_store_id=vector_store_id)

        # Extract only the IDs from the files
        file_ids = [file.id for file in vector_store_files.data]

        # Retrieve the details of each file and store them in a list
        files_details = []
        for file_id in file_ids:
            file_info = client.files.retrieve(file_id)  # Retrieve details for each file
            files_details.append({
                "id": file_info.id,
                "filename": file_info.filename
 
            })

        # Return the list of file IDs and corresponding file details
        return {
            "vector_store_id": vector_store_id,
            #"file_ids": file_ids,
            "files": files_details
        }

    except Exception as e:
        # Handle any exceptions and return a 400 HTTP error
        raise HTTPException(status_code=400, detail=str(e))

# POST: Create a new vector store
@app.post("/vector-store/create/")
async def create_vector_store(vector_store: VectorStoreCreate):
    try:
        new_vector_store = client.beta.vector_stores.create(name=vector_store.name)
        return {"message": "Vector store created successfully", "id": new_vector_store.id,"name":new_vector_store.name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error creating vector store: {e}")
    

@app.delete("/vector-store/delete/")
async def delete_vector_store(vector_store_id: str = Query(..., description="The ID of the Vector Store to be deleted")):
    try:
        # Call the client method to delete the vector store
        response = client.beta.vector_stores.delete(vector_store_id)
        
        # The delete method seems to return a VectorStoreDeleted object
        # If no exception is raised, we assume the deletion was successful
        return {"message": f"Vector Store {vector_store_id} deleted successfully"}
    except Exception as e:
        # If an exception occurs, raise an HTTP exception
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.delete("/files/delete")
async def delete_file(file_id: str = Query(..., description="The ID of the file to be deleted")):
    try:
        # Call the client method to delete the file
        response = client.files.delete(file_id)
        
        # Assuming response contains some status or result
        if response.get("status") == "success":
            return {"message": f"File {file_id} deleted successfully"}
        else:
            raise HTTPException(status_code=400, detail="File deletion failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vector-store/upload/")
async def upload_files_to_vector_store(
    vector_store_id: str = Form(..., description="ID of the vector store"),
    vector_store_name: str = Form(..., description="Name of the vector store"),
    assistant_id: str = Form(..., description="ID of the assistant"),
    assistant_name: str = Form(..., description="Name of the assistant"),
    files: List[UploadFile] = File(..., description="Multiple files to upload")
):
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded")

    try:
        # Get existing files in the vector store
        existing_files = client.beta.vector_stores.files.list(vector_store_id)
        existing_filenames = set(file_data.filename for file in existing_files if (file_data := client.files.retrieve(file.id)))

        file_streams = []
        uploaded_filenames = []
        for file in files:
            # Check if file with the same name already exists
            if file.filename in existing_filenames:
                raise HTTPException(status_code=400, detail=f"File '{file.filename}' already exists in the vector store '{vector_store_name}'  ")

            # Read file content
            file_content = await file.read()
            # Create a BytesIO object from the file content
            file_stream = io.BytesIO(file_content)
            # Set the name of the file
            file_stream.name = file.filename
            file_streams.append(file_stream)
            uploaded_filenames.append(file.filename)

        # Upload and poll the file batch
        file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store_id,
            files=file_streams
        )

        # Retrieve file details for uploaded files
        response = client.beta.vector_stores.files.list(vector_store_id)
        uploaded_files_info = []
        for file in response:
            file_id = file.id
            file_data = client.files.retrieve(file_id)
            if file_data.filename in uploaded_filenames:
                uploaded_files_info.append({
                    "id": file_id,
                    "filename": file_data.filename
                })

        # Update the assistant with the new vector store
        assistant = client.beta.assistants.update(
            assistant_id=assistant_id,
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
        )

        # Return the status, file counts of the batch, and request information
        return {
            "request_info": {
                "vector_store_id": vector_store_id,
                "vector_store_name": vector_store_name,
                "assistant_id": assistant_id,
                "assistant_name": assistant_name,
                "uploaded_files": uploaded_filenames
            },
            "response": {
                "status": file_batch.status,
                "file_counts": file_batch.file_counts,
                "uploaded_files_info": uploaded_files_info,
                "assistant_update": f"The vector store '{vector_store_name}' of Assistant '{assistant_name}' has been updated."
            }
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error uploading files: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
