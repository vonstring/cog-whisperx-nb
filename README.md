Cog implementation of [whisperX](https://github.com/m-bain/whisperX), a library that adds batch processing on top of whisper (and also faster-whisper), leading to very fast audio transcription. Using the [NbAiLabBeta/nb-whisper-large](https://huggingface.co/NbAiLabBeta/nb-whisper-large) model.

**Build Command:**

```bash
cog build -t whisperx
```

**Example Transcription Command:**

```bash
cog predict -i audio=@audio.m4a -i output_srt=True -i max_line_width=25 -i align_output=True
```

**Docker Command:**

```bash
docker run -d --gpus=all -p 5000:5000 --name whisperx whisperx
```

### Curl Example

```bash
curl -X POST -H "Content-Type: application/json" -d '{"input": {"audio":"https://host/audio.m4a", "align_output": true, "output_srt": true}}' http://localhost:5000/predictions
```

Read more about the HTTP API of Cog containers [here](https://github.com/replicate/cog/blob/main/docs/http.md)

### Options:

- `batch_size`: (int) Parallelization of input audio transcription.
- `align_output`: (bool) Enables word-level timing in the transcription.
- `only_text`: (bool) Returns only the text of the transcription.
- `output_srt`: (bool) Outputs transcription in SRT format.
- `output_vtt`: (bool) Outputs transcription in VTT format.
- `max_line_width` & `max_line_count`: (int) Customizes the formatting for SRT/VTT outputs.
- `highlight_words`: (bool) Highlights word timings in SRT output.
- `debug`: (bool) Outputs memory usage information for debugging.
