import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace  # noqa: E402

PROMPT = "<|im_start|>user\n你看这个<|im_end|>\n<|im_start|>assistant\n"


class Recorder:
    """An engine that can see pictures (like DeepSeekEngine)."""

    def __init__(self):
        self.calls = []

    def generate_json(self, prompt, max_tokens=200, temperature=0.7, media=None):
        self.calls.append({"prompt": prompt, "media": media})
        return {"reply": "Thought: 这张图有点意思\nResponse: 这图不错"}


class TextOnly:
    """An engine without a media parameter: pictures are simply not passed (old engines keep working)."""

    def __init__(self):
        self.calls = []

    def generate_json(self, prompt, max_tokens=200, temperature=0.7):
        self.calls.append({"prompt": prompt})
        return {"reply": "嗯"}


def bare_workspace(engine):
    gw = GlobalWorkspace.__new__(GlobalWorkspace)       # no database/body needed for generation
    gw.inference_engine = engine
    gw.last_verbal_confidence = None
    return gw


class TestPicturesInTheConversation(unittest.TestCase):
    def test_generate_response_hands_the_pictures_to_the_model(self):
        gw = bare_workspace(Recorder())
        media = [{"data": b"\xff\xd8photo", "mime": "image/jpeg"}]
        response = gw._generate_response(PROMPT, "开心", {"max_tokens": 60}, media=media)
        self.assertEqual(response, "这图不错")
        self.assertEqual(gw.inference_engine.calls[0]["media"], media)

    def test_text_only_engines_are_called_exactly_as_before(self):
        gw = bare_workspace(TextOnly())
        response = gw._generate_response(PROMPT, "开心", {"max_tokens": 60},
                                         media=[{"data": b"photo", "mime": "image/jpeg"}])
        self.assertEqual(response, "嗯")                   # media was not passed to a text-only engine
        self.assertEqual(len(gw.inference_engine.calls), 1)

    def test_no_pictures_means_no_media(self):
        gw = bare_workspace(Recorder())
        self.assertEqual(gw._generate_response(PROMPT, "开心", {"max_tokens": 60}), "这图不错")
        self.assertIsNone(gw.inference_engine.calls[0]["media"])


class TestSensesCaptionFallback(unittest.TestCase):
    """When the picture cannot ride along, the same DeepSeek API looks at it and returns one sentence."""

    class Engine:
        supports_images = True

        def __init__(self):
            self.media = []

        def generate_json(self, prompt, max_tokens=200, temperature=0.7, media=None):
            self.media.append(media)
            return {"reply": "一只橘猫趴在键盘上"}

    def senses(self):
        from napcat.senses import Senses
        s = Senses()
        s.caption_engine = self.Engine()
        s.fetch_image = lambda image: {"data": b"\xff\xd8photo", "mime": "image/jpeg"}
        return s

    def test_caption_uses_the_api_and_is_cached(self):
        s = self.senses()
        self.assertTrue(s.can_describe())
        self.assertEqual(s.see({"url": "u1"}), "一只橘猫趴在键盘上")
        self.assertEqual(s.see({"url": "u1"}), "一只橘猫趴在键盘上")     # looked at once, remembered
        self.assertEqual(len(s.caption_engine.media), 1)
        self.assertEqual(s.caption_engine.media[0][0], {"data": b"\xff\xd8photo", "mime": "image/jpeg"})

    def test_without_an_engine_she_cannot_describe(self):
        from napcat.senses import Senses
        s = Senses()
        s.fetch_image = lambda image: {"data": b"x", "mime": "image/jpeg"}
        self.assertFalse(s.can_describe())
        self.assertEqual(s.see({"url": "u1"}), "")

    def test_api_failure_is_fail_soft(self):
        from napcat.senses import Senses

        class Broken:
            supports_images = True

            def generate_json(self, prompt, **kwargs):
                raise RuntimeError("network down")

        s = Senses()
        s.caption_engine = Broken()
        s.fetch_image = lambda image: {"data": b"x", "mime": "image/jpeg"}
        self.assertEqual(s.see({"url": "u1"}), "")


if __name__ == "__main__":
    unittest.main()
