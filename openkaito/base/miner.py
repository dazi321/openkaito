import os
import time
import asyncio
import threading
import traceback
import bittensor as bt
import torch
from dotenv import load_dotenv
from openkaito.base.neuron import BaseNeuron
from openkaito.protocol import (
    DiscordSearchSynapse,
    SearchSynapse,
    SemanticSearchSynapse,
    StructuredSearchSynapse,
    TextEmbeddingSynapse,
)
from openkaito.utils.embeddings import openai_embeddings_tensor

# Load environment variables
load_dotenv()

class BaseMinerNeuron(BaseNeuron):
    """Optimized base class for Bittensor miners."""
    neuron_type: str = "MinerNeuron"

    def __init__(self):
        super().__init__(config=self.config())
        self.client = self._initialize_openai_client()
        self.axon = bt.axon(wallet=self.wallet, config=self.config)
        self._attach_handlers()
        self.last_sync_block = self.block - 1000
        self.should_exit = False
        self.is_running = False
        self.thread = None
        self.lock = asyncio.Lock()

    def _initialize_openai_client(self):
        return openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            organization=os.getenv("OPENAI_ORGANIZATION"),
            project=os.getenv("OPENAI_PROJECT"),
            max_retries=3,
        )

    def _attach_handlers(self):
        """Attach request processing functions."""
        handlers = [
            (self.forward_search, self.blacklist_search, self.priority_search),
            (self.forward_structured_search, self.blacklist_structured_search, self.priority_structured_search),
            (self.forward_semantic_search, self.blacklist_semantic_search, self.priority_semantic_search),
            (self.forward_discord_search, self.blacklist_discord_search, self.priority_discord_search),
            (self.forward_text_embedding, self.blacklist_text_embedding, self.priority_text_embedding),
        ]
        for forward, blacklist, priority in handlers:
            self.axon.attach(forward_fn=forward, blacklist_fn=blacklist, priority_fn=priority)
        bt.logging.info(f"Axon handlers attached.")

    async def forward(self, synapse: bt.Synapse) -> bt.Synapse:
        """Handles requests generically, routing them to appropriate handlers."""
        if isinstance(synapse, TextEmbeddingSynapse):
            return await self.forward_text_embedding(synapse)
        bt.logging.warning("Received unsupported synapse type.")
        return synapse

    async def forward_text_embedding(self, query: TextEmbeddingSynapse) -> TextEmbeddingSynapse:
        """Processes text embedding requests asynchronously."""
        query.results = openai_embeddings_tensor(
            self.client, query.texts, dimensions=query.dimensions, model="text-embedding-3-large"
        ).tolist()
        return query

    def print_info(self):
        """Logs miner status."""
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

    def run(self):
        """Main miner execution loop."""
        self.sync()
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()
        bt.logging.info(f"Miner starting at block: {self.block}")
        try:
            while not self.should_exit:
                while self.block - self.last_sync_block < self.config.neuron.epoch_length:
                    time.sleep(1)
                    if self.should_exit:
                        break
                self.sync()
                self.step += 1
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Miner stopped manually.")
            exit()
        except Exception as e:
            bt.logging.error(traceback.format_exc())

    def run_in_background_thread(self):
        """Starts the miner in a separate background thread."""
        if not self.is_running:
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True

    def stop_run_thread(self):
        """Stops the miner background thread."""
        if self.is_running:
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_run_thread()

    async def blacklist(self, synapse: bt.Synapse) -> typing.Tuple[bool, str]:
        """Determines if a request should be blacklisted."""
        if not synapse.dendrite.hotkey:
            return True, "Hotkey not provided"
        registered = synapse.dendrite.hotkey in self.metagraph.hotkeys
        if self.config.blacklist.allow_non_registered and not registered:
            return False, "Allowing un-registered hotkey"
        elif not registered:
            return True, f"Unrecognized hotkey {synapse.dendrite.hotkey}"
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if self.config.blacklist.force_validator_permit and not self.metagraph.validator_permit[uid]:
            return True, "Non-validator hotkey"
        if self.config.blacklist.validator_min_stake and self.metagraph.S[uid] < self.config.blacklist.validator_min_stake:
            return True, "Stake below minimum"
        return False, "Hotkey recognized!"

    async def priority(self, synapse: bt.Synapse) -> float:
        """Determines request priority based on stake."""
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return float(self.metagraph.S[caller_uid])

    async def blacklist_text_embedding(self, synapse: TextEmbeddingSynapse) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority_text_embedding(self, synapse: TextEmbeddingSynapse) -> float:
        return await self.priority(synapse)

    def save_state(self):
        pass

    def load_state(self):
        pass
