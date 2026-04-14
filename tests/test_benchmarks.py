from benchmarks.bench import make_jpg_b64, make_wav_b64, print_row


def test_benchmark_fixture_helpers_return_base64() -> None:
    assert make_wav_b64(0.01)
    assert make_jpg_b64()


def test_benchmark_print_row_uses_current_metrics(capsys) -> None:
    print_row(
        "Text only",
        {
            "text": "hello",
            "llm_time": 1.0,
            "tts_time": 0.5,
            "total_time": 1.7,
        },
    )

    assert "Text only" in capsys.readouterr().out
