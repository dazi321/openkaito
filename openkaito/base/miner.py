import asyncio
import threading
import time
import traceback
import typing
import bittensor as bt
import torch
from openkaito.base.neuron import BaseNeuron
from openkaito.protocol import (
    DiscordSearchSynapse, SearchSynapse, SemanticSearchSynapse,
    StructuredSearchSynapse, TextEmbeddingSynapse
)

class BaseMinerNeuron(BaseNeuron):
    """
    Base class for OpenKaito miners.
    Handles validator requests and manages mining operations.
    """

    neuron_type = "MinerNeuron"

    def __init__(self):
        super().__init__(config=self.config())
        self.axon = self.setup_axon()
        self.last_sync_block = self.block - 1000
        self.should_exit = False
        self.is_running = False
        self.lock = asyncio.Lock()
        self.thread = None

    def setup_axon(self):
        """Sets up the miner's Axon and attaches functions for request handling."""
        axon = bt.axon(wallet=self.wallet, config=self.config)
        bt.logging.info("Attaching forward functions to miner axon.")

        axon.attach(self.forward_search, self.blacklist_search, self.priority_search)
        axon.attach(self.forward_structured_search, self.blacklist_structured_search, self.priority_structured_search)
        axon.attach(self.forward_semantic_search, self.blacklist_semantic_search, self.priority_semantic_search)
        axon.attach(self.forward_discord_search, self.blacklist_discord_search, self.priority_discord_search)
        axon.attach(self.forward_text_embedding, self.blacklist_text_embedding, self.priority_text_embedding)

        bt.logging.info(f"Axon initialized: {axon}")
        return axon

    def run(self):
        """
        Runs the miner's event loop, handling metagraph sync and validator requests.
        """
        self.sync()
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()
        bt.logging.info(f"Miner running at block {self.block}")

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
            bt.logging.success("Miner stopped by user.")
        except Exception as e:
            bt.logging.error(traceback.format_exc())

    def run_in_background_thread(self):
        """Runs the miner in a separate thread."""
        if not self.is_running:
            bt.logging.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True

    def stop_run_thread(self):
        """Stops the miner's background execution."""
        if self.is_running:
            bt.logging.debug("Stopping miner thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False

    async def blacklist(self, synapse: bt.Synapse):
        """Implements basic blacklist logic."""
        return False, "Allowed"

    async def priority(self, synapse: bt.Synapse) -> float:
        """Sets request priority based on stake."""
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return float(self.metagraph.S[caller_uid])

    async def blacklist_text_embedding(self, synapse: TextEmbeddingSynapse):
        return await self.blacklist(synapse)

    async def priority_text_embedding(self, synapse: TextEmbeddingSynapse):
        return await self.priority(synapse)
