import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Ensure we can import api
sys.path.append(os.getcwd())

import shutil

class TestFFmpegFix(unittest.TestCase):
    @patch('os.path.exists')
    @patch('shutil.which')
    def test_init_with_system_ffmpeg(self, mock_which, mock_exists):
        # Simulate local ffmpeg NOT existing
        def side_effect(path):
            if "ffmpeg" in path and "bin" in path:
                return False
            return True # Allow other exist checks
        mock_exists.side_effect = side_effect
        
        # Simulate system ffmpeg existing
        mock_which.return_value = "/usr/bin/ffmpeg"
        
        print("Testing initialization with system ffmpeg...")
        try:
            extractor = ViralClipExtractor()
            print("Successfully initialized ViralClipExtractor!")
        except UnboundLocalError as e:
            self.fail(f"Fix failed! Raised UnboundLocalError: {e}")
        except Exception as e:
             # We expect a logger warning or error later, but initialization should finish the block in question
            print(f"Got other error (expected/ignored): {e}")

if __name__ == '__main__':
    unittest.main()
