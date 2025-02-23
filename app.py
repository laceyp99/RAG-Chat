import gradio as gr
import main

with gr.Blocks(css="""
    .center-title { text-align: center; font-size: 3em; }
    .accordion-header { font-size: 16px; font-weight: bold; }
""") as demo:
    gr.Markdown("<h1 class='center-title'>RAG Chatbot 🤖</h1>")
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Document Upload/Loading 📄")
            file_upload = gr.File(label="Upload Files", file_count="multiple", type="filepath")
            # web_urls = gr.Textbox(
            #     lines=4, 
            #     label="Enter Web URLs (optional)", 
            #     placeholder="Link 1 (https://example.com/page1),\nLink 2 (https://example.com/page2)"
            # )
            load_button = gr.Button("Load Documents ⚙️")
            load_status = gr.Textbox(label="Load Status", interactive=False)
            with gr.Accordion("Advanced Parameters ⚙️", open=False):
                chunk_size_slider = gr.Slider(
                    minimum=500, 
                    maximum=2000, 
                    step=100, 
                    value=1000, 
                    label="Chunk Size"
                )
                chunk_overlap_slider = gr.Slider(
                    minimum=0, 
                    maximum=500, 
                    step=50, 
                    value=200, 
                    label="Chunk Overlap"
                )
                retrieval_k_slider = gr.Slider(
                    minimum=1, 
                    maximum=10, 
                    step=1, 
                    value=2, 
                    label="Retrieval k"
                )
        with gr.Column():
            gr.Markdown("## Chat with RAG Bot 💬")
            query_text = gr.Textbox(label="Enter your Prompt", placeholder="Ask your question here...")
            temperature_slider = gr.Slider(
                minimum=0, 
                maximum=2, 
                value=0.0, 
                step=0.1, 
                label="Temperature 🌡️"
            )
            query_button = gr.Button("Generate Response 🚀")
            answer_text = gr.Textbox(label="Response", interactive=False)
    
    load_button.click(
        fn=main.build_chain, 
        inputs=[file_upload, chunk_size_slider, chunk_overlap_slider, retrieval_k_slider],
        outputs=load_status
    )
    query_button.click(
        fn=main.process_query, 
        inputs=[query_text, temperature_slider], 
        outputs=answer_text
    )

demo.launch()