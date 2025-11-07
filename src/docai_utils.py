from google.cloud import documentai_v1 as documentai

def process_document_with_processor(project_id, location, processor_id, content, mime_type):
    client = documentai.DocumentProcessorServiceClient()
    name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

    raw_document = documentai.RawDocument(
        content=content,
        mime_type=mime_type
    )

    request = documentai.ProcessRequest(
        name=name,
        raw_document=raw_document
    )

    response = client.process_document(request=request)
    return response.document

