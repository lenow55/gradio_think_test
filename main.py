import gradio as gr
from gradio import ChatMessage
from collections.abc import Iterator
import openai
from openai.types.chat import ChatCompletionChunk

model = "Qwen/Qwen3-8B-FP8"

client = openai.OpenAI(
    base_url="http://localhost:8953/v1",
    api_key="NYXV2sS3PLDYLbC",
)


def stream_openai_response(
    user_message: str, messages: list[ChatMessage]
) -> Iterator[list[ChatMessage]]:
    """
    Streams responses from OpenAI model with thinking process simulation
    """
    # Initialize response from OpenAI
    response: openai.Stream[ChatCompletionChunk] = client.chat.completions.create(  # pyright: ignore[reportUnknownVariableType,reportCallIssue]
        model=model,
        messages=messages + [{"role": "user", "content": user_message}],  # pyright: ignore[reportArgumentType]
        stream=True,
    )

    # Initialize buffers
    thought_buffer = ""
    response_buffer = ""

    # Add initial thinking message
    messages.append(
        ChatMessage(
            role="assistant",
            content="",
            metadata={
                "title": "⏳Thinking: *The thoughts produced by the OpenAI model are simulated"
            },
        ),
    )
    messages.append(
        ChatMessage(
            role="assistant",
            content="",
        )
    )
    if not isinstance(response, openai.Stream):
        raise RuntimeError()

    # тут код падает
    for chunk in response:
        if not isinstance(chunk, ChatCompletionChunk):
            continue

        reasoning = None
        if isinstance(chunk.choices[0].delta.__pydantic_extra__, dict):
            reasoning = chunk.choices[0].delta.__pydantic_extra__.get("reasoning")
            reasoning = chunk.choices[0].delta.__pydantic_extra__.get(
                "reasoning_content"
            )
        if isinstance(reasoning, str):
            # Simulate thinking process
            thought_buffer += reasoning
            messages[-2] = ChatMessage(
                role="assistant",
                content=thought_buffer,
                metadata={
                    "title": "⏳Thinking: *The thoughts produced by the OpenAI model are simulated"
                },
            )
        if isinstance(chunk.choices[0].delta.content, str):
            # Stream final response
            content = chunk.choices[0].delta.content
            response_buffer += str(content)
            messages[-1] = ChatMessage(role="assistant", content=response_buffer)

        yield messages


def user_message(msg: str, chatbot: list[ChatMessage]):
    """Add user message to chat"""
    chatbot.append(ChatMessage(role="user", content=msg))
    return chatbot


with gr.Blocks() as demo:
    gr.Markdown("# Chat with OpenAI and See its Thoughts 💭")

    chatbot = gr.Chatbot(
        type="messages",
        label="OpenAI Chatbot",
        render_markdown=True,
    )

    input_box = gr.Textbox(
        lines=1,
        label="Chat Message",
        placeholder="Type your message here and press Enter...",
    )

    msg_store = gr.State("")  # Store for preserving user message

    _ = (
        input_box.submit(
            lambda msg: (msg, msg, ""),  # Store message and clear input
            inputs=[input_box],
            outputs=[msg_store, input_box, input_box],
            queue=False,
        )
        .then(
            user_message,  # Add user message to chat
            inputs=[msg_store, chatbot],
            outputs=chatbot,
            queue=False,
        )
        .then(
            stream_openai_response,  # Generate and stream response
            inputs=[msg_store, chatbot],
            outputs=chatbot,
        )
    )

demo.launch()
