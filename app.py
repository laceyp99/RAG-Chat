import gradio as gr
import main

with gr.Blocks() as demo:
    gr.Markdown("# RAG Chatbot")
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Document Upload/Loading")
            file_upload = gr.File(label="Upload Files", file_count="multiple", type="filepath")
            # web_urls = gr.Textbox(
            #     lines=4, 
            #     label="Enter Web URLs (optional)", 
            #     placeholder="Link 1 (https://example.com/page1),\nLink 2 (https://example.com/page2)"
            # )
            load_button = gr.Button("Load Documents")
            load_status = gr.Textbox(label="Load Status", interactive=False)
        with gr.Column():
            gr.Markdown("## Chat with RAG Bot")
            query_text = gr.Textbox(label="Enter your Prompt", placeholder="Ask your question here...")
            query_button = gr.Button("Generate Response")
            answer_text = gr.Textbox(label="Response", interactive=False)
    
    # Pass both the web URLs and uploaded files to main.process_document_loading.
    load_button.click(
        fn=main.build_chain, 
        inputs=file_upload, # create a list with web_urls and file_upload for web scraping
        outputs=load_status
    )
    query_button.click(
        fn=main.process_query, 
        inputs=query_text, 
        outputs=answer_text
    )

demo.launch()