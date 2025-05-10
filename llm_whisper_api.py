import click
import httpx
import io
import llm
from pathlib import Path


@llm.hookimpl
def register_commands(cli):
    @cli.command()
    @click.argument("audio_file", type=click.File("rb"))
    @click.option("api_key", "--key", help="API key to use")
    @click.option(
        "-m", "--model",
        type=click.Choice(["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"]),
        default="whisper-1",
        help="Model to use for transcription",
    )
    def whisper_api(audio_file, api_key, model):
        """
        Run transcriptions using the OpenAI Whisper API

        Usage:

        \b
            llm whisper-api audio.mp3 > output.txt
            cat audio.mp3 | llm whisper-api - > output.txt
            llm whisper-api -m gpt-4o-transcribe audio.mp3 > output.txt
        """
        # Read the entire content into memory first
        audio_content = audio_file.read()
        audio_file.close()
        audio_stream = io.BytesIO(audio_content)
        suffix = Path(audio_file.name).suffix
        audio_stream.name = f"audio{suffix}"  # OpenAI API requires a filename, or 400 error

        key = llm.get_key(api_key, "openai")
        if not key:
            raise click.ClickException("OpenAI API key is required")
        try:
            click.echo(transcribe(audio_stream, key, model))
        except httpx.HTTPError as ex:
            raise click.ClickException(str(ex))


def transcribe(audio_stream: io.BytesIO, api_key: str, model: str) -> str:
    """
    Transcribe audio content using OpenAI's Whisper API.

    Args:
        audio_stream (io.BytesIO): The audio content as stream
        api_key (str): OpenAI API key
        model (str): The model name to use for transcription

    Returns:
        str: The transcribed text

    Raises:
        httpx.RequestError: If the API request fails
    """
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}

    files = {"file": audio_stream}
    data = {"model": model, "response_format": "json"}

    with httpx.Client() as client:
        response = client.post(url, headers=headers, files=files, data=data)
        response.raise_for_status()
        return response.json()["text"].strip()
