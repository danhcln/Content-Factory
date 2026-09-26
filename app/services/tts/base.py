from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseTTSProvider(ABC):
    @abstractmethod
    def synthesize(self, text: str, output_path: str, voice_name: Optional[str] = None) -> Dict[str, Any]:
        """Synthesize text to speech and save as audio WAV file."""
        pass

    @abstractmethod
    def get_available_voices(self) -> List[Dict[str, str]]:
        """List available voices for this provider."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether local models and dependencies are operational."""
        pass
