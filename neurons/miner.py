import os
import time
import asyncio
import openai
import bittensor as bt
from dotenv import load_dotenv
from openkaito.base.miner import BaseMinerNeuron
from openkaito.utils.embeddings import openai_embeddings_tensor
from openkaito.utils.version import compare_version, get_version
from openkaito.protocol import TextEmbeddingSynapse

# Load environment variables
load_dotenv()

class Miner(BaseMinerNeuron):
    """
    OpenKaito Miner optimized for text embedding requests.
    """
    def __init__(self):
        super().__init__()
        self.client = openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            organization=os.getenv("OPENAI_ORGANIZATION"),
            project=os.getenv("OPENAI_PROJECT"),
            max_retries=3,
        )

    async def forward_text_embedding(self, query: TextEmbeddingSynapse) -> TextEmbeddingSynapse:
        """Processes text embedding requests asynchronously."""
        query.results = openai_embeddings_tensor(
            self.client, query.texts, dimensions=query.dimensions, model="text-embedding-3-large"
        ).tolist()
        return query

    def print_info(self):
        """Logs miner status in real time."""
        try:
            metagraph = self.metagraph
            uid = metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            log = (
                f"Epoch:{self.step} | UID:{uid} | Stake:{metagraph.S[uid]:.6f} | "
                f"Rank:{metagraph.R[uid]:.4f} | Trust:{metagraph.T[uid]:.4f} | "
                f"Consensus:{metagraph.C[uid]:.4f} | Incentive:{metagraph.I[uid]:.6f} | "
                f"Emission:{metagraph.E[uid]:.6f}"
            )
            bt.logging.info(log)
        except Exception as e:
            bt.logging.warning(f"Error printing miner info: {e}")

    def check_version(self, query):
        """Checks for version mismatches and warns if needed."""
        if query.version and compare_version(query.version, get_version()) > 0:
            bt.logging.warning(
                f"Received request with newer version {query.version}. Consider updating your miner."
            )

if __name__ == "__main__":
    with Miner() as miner:
        print(f"Miner hotkey: {miner.wallet.hotkey.ss58_address}")
        asyncio.run(asyncio.sleep(60))
        while True:
            miner.print_info()
            time.sleep(15)
