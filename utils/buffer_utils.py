import collections
import jax
import logging
from typing import Sequence, Tuple
import numpy as np
from flax.core import FrozenDict
import optax

# basic batch
Batch = collections.namedtuple(
    "Batch",
    ["observations", "actions", "rewards", "discounts", "next_observations"])


FinetuneBatch = collections.namedtuple(
    "FinetuneBatch",
    ["observations", "actions", "rewards", "discounts", "next_observations", "embeddings", "feedback_embeddings", "successes", "keyframe_weights", "next_embeddings"])

MaskBatch = collections.namedtuple(
    "MaskBatch",
    ["observations", "actions", "rewards", "discounts", "next_observations", "embeddings", "feedback_embeddings", "masks", "successes", "keyframe_weights", "next_embeddings"])


VLMBatch = collections.namedtuple(
    "VLMBatch",
    ["observations", "actions", "rewards", "vlm_rewards", "discounts", "next_observations"])


EmbeddingBatch = collections.namedtuple(
    "EmbeddingBatch",
    ["pos_embeddings", "neg_embeddings", "lag_embeddings"])

FeedbackBatch = collections.namedtuple(
    "FeedbackBatch",
    ["image_embeddings", "feedback_embeddings"]
)

class ReplayBuffer:

    def __init__(self, obs_dim: int, act_dim: int, max_size: int = int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.observations = np.zeros((max_size, obs_dim))
        self.actions = np.zeros((max_size, act_dim))
        self.next_observations = np.zeros((max_size, obs_dim))
        self.rewards = np.zeros(max_size)
        self.discounts = np.zeros(max_size)

    def add(self,
            observation: np.ndarray,
            action: np.ndarray,
            next_observation: np.ndarray,
            reward: float,
            done: float):
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.next_observations[self.ptr] = next_observation
        self.rewards[self.ptr] = reward
        self.discounts[self.ptr] = 1 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = Batch(observations=self.observations[idx],
                      actions=self.actions[idx],
                      rewards=self.rewards[idx],
                      discounts=self.discounts[idx],
                      next_observations=self.next_observations[idx])
        return batch

    def save(self, fname: str):
        np.savez(fname,
                 observations=self.observations,
                 actions=self.actions,
                 next_observations=self.next_observations,
                 rewards=self.rewards,
                 discounts=self.discounts)


class VLMBuffer:

    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 max_size: int = int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.observations = np.zeros((max_size, obs_dim))
        self.actions = np.zeros((max_size, act_dim))
        self.next_observations = np.zeros((max_size, obs_dim))
        self.vlm_rewards = np.zeros(max_size)
        self.rewards = np.zeros(max_size)
        self.discounts = np.zeros(max_size)

    def add(self,
            observation: np.ndarray,
            action: np.ndarray,
            next_observation: np.ndarray,
            vlm_reward: float,
            reward: float,
            done: float):
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.next_observations[self.ptr] = next_observation
        self.vlm_rewards[self.ptr] = vlm_reward
        self.rewards[self.ptr] = reward
        self.discounts[self.ptr] = 1 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, self.size, size=batch_size)
        rewards = self.rewards[idx]
        vlm_rewards = self.vlm_rewards[idx]
        batch = VLMBatch(observations=self.observations[idx],
                       actions=self.actions[idx],
                       rewards=rewards,
                       vlm_rewards=vlm_rewards,
                       discounts=self.discounts[idx],
                       next_observations=self.next_observations[idx])
        return batch


class DistanceBuffer:

    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 emb_dim: int = 1024,
                 fb_dim:int = 768,             # for GPT 768, for LIV 1024
                 max_size: int = int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.fdb_ptr = 0                # pointer to position of feedback -> Monaf
        self.successes = np.zeros(max_size, dtype=np.float32) # New: To store success labels

        self.observations = np.zeros((max_size, obs_dim))
        self.actions = np.zeros((max_size, act_dim))
        self.rewards = np.zeros(max_size)
        self.next_observations = np.zeros((max_size, obs_dim))
        self.discounts = np.zeros(max_size)
        self.embeddings = np.zeros((max_size, emb_dim))
        self.next_embeddings = np.zeros((max_size, emb_dim))
        self.distances = np.zeros((max_size))
        self.feedback_embeddings = np.zeros((max_size, fb_dim), dtype=np.float32) # Feedback buffer
        self.keyframe_weights = np.zeros(max_size, dtype=np.float32)


    def add(self,
            observation: np.ndarray,
            action: np.ndarray,
            next_observation: np.ndarray, 
            reward: float,
            done: float,
            embedding: np.ndarray,
            success: float, # New: Pass the success status of the step
            distance: float = 0):

        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.next_observations[self.ptr] = next_observation
        self.rewards[self.ptr] = reward
        self.discounts[self.ptr] = 1 - done
        self.embeddings[self.ptr] = embedding
        self.successes[self.ptr] = success # Store the success label
        self.distances[self.ptr] = distance

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_episode(self,
                    transitions: Sequence[Tuple[
                        np.ndarray,  # obs
                        np.ndarray,  # action
                        np.ndarray,  # next_obs
                        float,       # reward
                        float,       # done
                        np.ndarray,  # embedding
                        float,       # success
                        float        # distance
                    ]],
                    feedback_embedding: np.ndarray,
                    fb_weights: np.array):
        """
        Bulk-insert an entire episode of length N into the ring buffer,
        stamping the same feedback_embedding for each transition.
        """
        N = len(transitions)
        idxs = (self.ptr + np.arange(N)) % self.max_size

        # unpack into batched arrays
        obs_batch      = np.stack([t[0] for t in transitions], axis=0)
        act_batch      = np.stack([t[1] for t in transitions], axis=0)
        next_obs_batch = np.stack([t[2] for t in transitions], axis=0)
        reward_batch   = np.array([t[3] for t in transitions], dtype=np.float32)
        done_batch     = np.array([t[4] for t in transitions], dtype=np.float32)
        emb_batch      = np.stack([t[5] for t in transitions], axis=0)
        success_batch  = np.array([t[6] for t in transitions], dtype=np.float32)
        dist_batch     = np.array([t[7] for t in transitions], dtype=np.float32)
        emb_next_batch = np.concatenate([emb_batch[1:], emb_batch[-1:]], axis=0)  # shift by 1, last repeats

        # broadcast feedback to (N, fb_dim)
        fb_batch = np.broadcast_to(feedback_embedding, (N, feedback_embedding.shape[-1]))
        self.keyframe_weights[idxs] = fb_weights

        # one‐shot writes
        self.observations[idxs]        = obs_batch
        self.actions[idxs]             = act_batch
        self.next_observations[idxs]   = next_obs_batch
        self.rewards[idxs]             = reward_batch
        self.discounts[idxs]           = 1.0 - done_batch
        self.embeddings[idxs]          = emb_batch
        self.next_embeddings[idxs]     = emb_next_batch
        self.successes[idxs]           = success_batch
        self.distances[idxs]           = dist_batch
        self.feedback_embeddings[idxs] = fb_batch

        # advance ring pointers
        self.ptr  = (self.ptr + N) % self.max_size
        self.size = min(self.size + N, self.max_size)

    def sample_with_mask(self, batch_size: int, l2_margin: float = 0.05) -> Batch:
        idx = np.random.randint(0, self.size, size=batch_size)
        distance = self.distances[idx]

        l2_delta = distance.reshape(-1, 1) - distance.reshape(1, -1)
        masks = (l2_delta < -l2_margin).astype(np.float32)

        batch = MaskBatch(observations=self.observations[idx],
                          actions=self.actions[idx],
                          rewards=self.rewards[idx],
                          discounts=self.discounts[idx],
                          next_observations=self.next_observations[idx],
                          embeddings=self.embeddings[idx],
                          feedback_embeddings=self.feedback_embeddings[idx],  # feedback in batch
                          masks=masks,
                          successes=self.successes[idx], # Sample the success labels
                          keyframe_weights= self.keyframe_weights[idx],
                          next_embeddings=self.next_embeddings[idx])

        return batch

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = FinetuneBatch(observations=self.observations[idx],
                              actions=self.actions[idx],
                              rewards=self.rewards[idx],
                              discounts=self.discounts[idx],
                              next_observations=self.next_observations[idx],
                              embeddings=self.embeddings[idx],
                              feedback_embeddings=self.feedback_embeddings[idx], # ← feedback here
                              successes=self.successes[idx], # Sample the success labels)    
                              keyframe_weights = self.keyframe_weights[idx],
                              next_embeddings=self.next_embeddings[idx])

        return batch


class EmbeddingBuffer:
    def __init__(self,
                 emb_dim: int,
                 gap: int = 10,
                 max_size: int = int(1e5)):
        self.gap = gap
        self.max_size = max_size

        self.pos_ptr = 0
        self.pos_size = 0
        self.pos_embeddings = np.zeros((max_size, emb_dim))

        self.neg_ptr = 0
        self.neg_size = 0
        self.neg_embeddings = np.zeros((max_size, emb_dim))

        self.valid_ptr = 0
        self.valid_size = 0
        self.valid_idxes = np.zeros(max_size, dtype=np.int32)

    def add(self,
            embedding: np.ndarray,
            success: bool = False,
            valid: bool = False):
        if success:
            self.pos_embeddings[self.pos_ptr] = embedding
            if valid:
                self.valid_idxes[self.valid_ptr] = self.pos_ptr
                self.valid_ptr = (self.valid_ptr + 1) % self.max_size
                self.valid_size = min(self.valid_size + 1, self.max_size)
            self.pos_ptr = (self.pos_ptr + 1) % self.max_size
            self.pos_size = min(self.pos_size + 1, self.max_size)
        else:
            self.neg_embeddings[self.neg_ptr] = embedding
            self.neg_ptr = (self.neg_ptr + 1) % self.max_size
            self.neg_size = min(self.neg_size + 1, self.max_size)

    def sample(self, batch_size):
        neg_idx = np.random.randint(0, self.neg_size, size=batch_size)
        valid_idx = np.random.randint(0, self.valid_size, size=batch_size)
        pos_idx = self.valid_idxes[valid_idx]
        lag_idx = (pos_idx - self.gap) % self.valid_size

        pos_embeddings = self.pos_embeddings[pos_idx]
        lag_embeddings = self.pos_embeddings[lag_idx]
        neg_embeddings = self.neg_embeddings[neg_idx]
        return EmbeddingBatch(pos_embeddings=pos_embeddings,
                              lag_embeddings=lag_embeddings,
                              neg_embeddings=neg_embeddings)

    def save(self, fdir):
        np.savez(fdir,
                 pos_embeddings=self.pos_embeddings,
                 neg_embeddings=self.neg_embeddings,
                 pos_ptr=self.pos_ptr,
                 pos_size=self.pos_size,
                 neg_ptr=self.neg_ptr,
                 neg_size=self.neg_size,
                 valid_ptr=self.valid_ptr,
                 valid_size=self.valid_size,
                 valid_idxes=self.valid_idxes)
