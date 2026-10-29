"""
Tests for export_to_ollama.py: Modelfile generation and Ollama deployment guide.
Compatible with both pytest and unittest.
"""

import os
import shutil
import tempfile
import unittest

from export_to_ollama import create_ollama_modelfile, generate_deployment_guide


class TestOllamaExport(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_lora_ollama_")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_create_ollama_modelfile(self):
        modelfile_path = os.path.join(self.temp_dir, "Modelfile")
        merged_dir = os.path.join(self.temp_dir, "merged_hf_model")
        os.makedirs(merged_dir, exist_ok=True)

        created_path = create_ollama_modelfile(
            merged_model_dir=merged_dir,
            output_modelfile_path=modelfile_path,
            model_name="password-gen",
        )

        self.assertTrue(os.path.exists(created_path))
        with open(created_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("FROM", content)
        self.assertIn("PARAMETER temperature 0.0", content)
        self.assertIn("PARAMETER stop", content)
        self.assertIn("TEMPLATE", content)
        self.assertIn("Input:", content)

    def test_generate_deployment_guide(self):
        guide_path = os.path.join(self.temp_dir, "OLLAMA_README.md")
        modelfile_path = os.path.join(self.temp_dir, "Modelfile")
        merged_dir = os.path.join(self.temp_dir, "merged_hf_model")

        generate_deployment_guide(
            merged_dir=merged_dir,
            modelfile_path=modelfile_path,
            guide_path=guide_path,
        )

        self.assertTrue(os.path.exists(guide_path))
        with open(guide_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("ollama create password-gen", content)
        self.assertIn("llama.cpp", content)
        self.assertIn("convert_hf_to_gguf.py", content)


if __name__ == "__main__":
    unittest.main()
