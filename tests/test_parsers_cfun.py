"""Tests for CFun extraction and cleaning in parsers.parse_cfun."""

from data_preprocessing.parsers import parse_cfun


def test_parse_cfun_instruction_routing_and_concat(monkeypatch, tmp_path):
    """CFun parser routes fields by instruction and concatenates first-sentence samples."""
    records = [
        {
            "instruction": "生成一个主题为校园的笑话",
            "input": "",
            "output": "这是一个主题笑话输出文本",
        },
        {
            "instruction": (
                "以下是一段文本，请分析它是否具有幽默性。幽默性指该文本是否可能引起读者发笑，"
                "或通过语言技巧（如双关语、讽刺、夸张、荒诞或逻辑上的意外）营造幽默效果。只需要输出“幽默”或“不幽默”。"
            ),
            "input": "判定任务中的输入笑话",
            "output": "幽默",
        },
        {
            "instruction": "我将给你笑话的第一句话，请你生成整个笑话。笑话的第一句话如下：",
            "input": "这是笑话的第一句话，",
            "output": "这是后半句并且长度足够。",
        },
        {
            "instruction": "不支持的指令",
            "input": "应被丢弃",
            "output": "应被丢弃",
        },
    ]

    def _fake_load_dataset(*args, **kwargs):
        return {"train": records}

    monkeypatch.setattr("data_preprocessing.parsers.datasets.load_dataset", _fake_load_dataset)

    ds = parse_cfun(tmp_path / "cfun_cache")
    texts = ds["text"]

    assert "这是一个主题笑话输出文本" in texts
    assert "判定任务中的输入笑话" in texts
    assert "这是笑话的第一句话，这是后半句并且长度足够。" in texts
    assert all("应被丢弃" not in t for t in texts)
    assert all(lang == "zh" for lang in ds["lang"])
    assert all(source == "cfun" for source in ds["source"])


def test_parse_cfun_cleaning_length_and_dedup(monkeypatch, tmp_path):
    """Parser removes labels, filters by length, and deduplicates cleaned text."""
    records = [
        {
            "instruction": "生成一个关键词为测试的笑话",
            "input": "",
            "output": "标题：内容：这是一个足够长的重复笑话文本",
        },
        {
            "instruction": "生成一个关键词为测试的笑话",
            "input": "",
            "output": "这是一个足够长的重复笑话文本",
        },
        {
            "instruction": "生成一个主题为长度的笑话",
            "input": "",
            "output": "短",
        },
        {
            "instruction": "生成一个主题为长度的笑话",
            "input": "",
            "output": "很长" * 260,
        },
        {
            "instruction": "生成一个主题为无效字符的笑话",
            "input": "",
            "output": "！！！？？？",
        },
    ]

    def _fake_load_dataset(*args, **kwargs):
        return {"train": records}

    monkeypatch.setattr("data_preprocessing.parsers.datasets.load_dataset", _fake_load_dataset)

    ds = parse_cfun(tmp_path / "cfun_cache")
    texts = ds["text"]

    assert len(texts) == 1
    assert texts[0] == "这是一个足够长的重复笑话文本"
