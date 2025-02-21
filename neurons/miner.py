import os
import openai
import asyncio
import bittensor as bt
from dotenv import load_dotenv
from openkaito.base.miner import BaseMinerNeuron
from openkaito.protocol import TextEmbeddingSynapse
from openkaito.utils.embeddings import openai_embeddings_tensor
from openkaito.utils.version import compare_version, get_version

# Load environment variables
load_dotenv()

class Miner(BaseMinerNeuron):
    """
    Custom miner neuron for the OpenKaito subnet.
    Handles text embedding requests using OpenAI's API.
    """

    def __init__(self):
        super(Miner, self).__init__()
        self.client = self.init_openai_client()

    def init_openai_client(self):
        """Initialize OpenAI client with environment variables."""
        return openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            organization=os.getenv("OPENAI_ORGANIZATION"),
            project=os.getenv("OPENAI_PROJECT"),
            max_retries=2,
            timeout=5,
        )

    async def forward(self, synapse: bt.Synapse) -> bt.Synapse:
        """
        Generic forward function that directs requests to the correct method.
        """
        if isinstance(synapse, TextEmbeddingSynapse):
            return await self.forward_text_embedding(synapse)
        else:
            bt.logging.warning(f"Unsupported synapse type: {type(synapse)}")
            return synapse  # Return unchanged if unsupported.

    async def forward_text_embedding(self, query: TextEmbeddingSynapse) -> TextEmbeddingSynapse:
        """
        Handles incoming text embedding requests.
        """
        try:
            embeddings = openai_embeddings_tensor(
                self.client, query.texts, dimensions=query.dimensions, model="text-embedding-3-large"
            )
            query.results = embeddings.tolist()
        except Exception as e:
            bt.logging.error(f"Error generating embeddings: {e}")
            query.results = []
        return query

    def print_info(self):
        """Logs miner's current state and statistics."""
        try:
            self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            log_msg = (
                f"Miner | Epoch: {self.step} | UID: {self.uid} | Block: {self.block} | "
                f"Stake: {self.metagraph.S[self.uid]:.4f} | Rank: {self.metagraph.R[self.uid]:.4f} | "
                f"Trust: {self.metagraph.T[self.uid]:.4f} | Consensus: {self.metagraph.C[self.uid]:.4f} | "
                f"Incentive: {self.metagraph.I[self.uid]:.4f} | Emission: {self.metagraph.E[self.uid]:.4f}"
            )
            bt.logging.info(log_msg)
        except Exception as e:
            bt.logging.error(f"Error printing miner info: {e}")

    def check_version(self, query):
        """Warns if the received request version is newer than the current miner version."""
        if query.version and compare_version(query.version, get_version()) > 0:
            bt.logging.warning(
                f"Received request version {query.version} is newer than miner version {get_version()}."
                " Consider updating the repo and restarting the miner."
            )

async def run_miner(miner):
    """
    Runs the miner asynchronously, logging information at regular intervals.
    """
    await asyncio.sleep(60)  # Wait for 60 seconds before starting
    while True:
        miner.print_info()
        await asyncio.sleep(5)  # Faster updates

if __name__ == "__main__":
    miner = Miner()
    print(f"My Miner hotkey: {miner.wallet.hotkey.ss58_address}")
    asyncio.run(run_miner(miner))
