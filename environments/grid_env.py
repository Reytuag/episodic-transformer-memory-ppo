from typing import Tuple
from gymnasium import spaces
from PIL import Image
from PIL import ImageDraw
import numpy as np

import jax
import jax.numpy as jnp
from jax import random
from flax.struct import dataclass

from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask

SIZE_GRID = 3
AGENT_VIEW = 2


@dataclass
class AgentState(object):
    posx: jnp.int32
    posy: jnp.int32
    inventory: jnp.int32


@dataclass
class State(TaskState):
    obs: jnp.ndarray
    last_action: jnp.ndarray
    reward: jnp.ndarray
    state: jnp.ndarray
    agent: AgentState
    steps: jnp.int32
    permutation_recipe: jnp.ndarray
    key: jnp.ndarray
    grid_color: jnp.ndarray
    immo: jnp.int32


def get_obs(state: jnp.ndarray, posx: jnp.int32, posy: jnp.int32, grid_color: jnp.ndarray) -> jnp.ndarray:
    state_color = (jnp.expand_dims(state[:, :, 1:], axis=-1) * grid_color[:, :, 1:]).sum(axis=2)

    obs = jnp.ravel(jax.lax.dynamic_slice(
        jnp.pad(state_color, ((AGENT_VIEW, AGENT_VIEW), (AGENT_VIEW, AGENT_VIEW), (0, 0)), constant_values=1),
        (posx - AGENT_VIEW + AGENT_VIEW, posy - AGENT_VIEW + AGENT_VIEW, 0),
        (2 * AGENT_VIEW + 1, 2 * AGENT_VIEW + 1, 3)))
    return obs


def get_init_state_fn(key: jnp.ndarray) -> jnp.ndarray:
    grid = jnp.zeros((SIZE_GRID, SIZE_GRID, 6))
    posx, posy = (1, 1)
    grid = grid.at[posx, posy, 0].set(1)
    next_key, key = random.split(key)
    # pos_obj=jax.random.randint(key,(6,2),0,SIZE_GRID)
    # pos_obj=jnp.array([[0,1],[1,2],[2,1]])
    # grid=grid.at[pos_obj[0,0],pos_obj[0,1],1].add(1)
    # grid=grid.at[pos_obj[1,0],pos_obj[1,1],2].add(1)
    # grid=grid.at[pos_obj[2,0],pos_obj[2,1],3].add(1)
    # grid=grid.at[pos_obj[3,0],pos_obj[3,1],1].add(1)
    # grid=grid.at[pos_obj[4,0],pos_obj[4,1],2].add(1)
    # grid=grid.at[pos_obj[5,0],pos_obj[5,1],3].add(1)

    # next_key, key = random.split(next_key)
    # perm=jax.random.permutation(key,3)+1

    # grid=grid.at[pos_obj[0,0],pos_obj[0,1],perm[0]].add(1)
    # grid=grid.at[pos_obj[1,0],pos_obj[1,1],perm[1]].add(1)
    # grid=grid.at[pos_obj[2,0],pos_obj[2,1],perm[2]].add(1)

    # pos_ax=jax.random.randint(key,(3,),0,SIZE_GRID)

    next_key, key = random.split(next_key)
    perm = jax.random.randint(key, (3,), 0, SIZE_GRID)
    # pos_obj=jnp.array([[perm[0],0],[perm[1],1],[perm[2],2]])
    next_key, key = random.split(next_key)
    perm2 = jax.random.permutation(key, 3)

    pos_obj = jnp.array([[perm[0], perm2[0]], [perm[1], perm2[1]], [perm[2], perm2[2]]])

    # next_key, key = random.split(next_key)
    # perm=jax.random.permutation(key,3)+1

    grid = grid.at[pos_obj[0, 0], pos_obj[0, 1], 1].add(1)
    grid = grid.at[pos_obj[1, 0], pos_obj[1, 1], 2].add(1)
    grid = grid.at[pos_obj[2, 0], pos_obj[2, 1], 3].add(1)

    return (grid)


def test_recipes(items, recipes):
    recipe_done = jnp.where(items[recipes[0]] * items[recipes[1]] > 0, jnp.array([recipes[0], recipes[1], 4]),
                            jnp.zeros(3, jnp.int32))
    # recipe_done=jnp.where(items[recipes[2]]*items[4]>0,jnp.array([recipes[2],4,5]),recipe_done)
    product = recipe_done[2]
    reward = jnp.select([product == 0, product == 4], [0.0, 1.])
    return recipe_done, reward


def drop(grid, posx, posy, inventory, recipes):
    # vanilla drop

    # grid=grid.at[posx,posy,inventory].add(1)
    # inventory=0
    # cost=-0.
    # test recipe
    # recipe_done,reward=jax.lax.cond(grid[posx,posy,1:].sum()==2,test_recipes,lambda x,y:(jnp.zeros(3,jnp.int32),0.),*(grid[posx,posy,:],recipes))
    # grid=jnp.where(recipe_done[2]>0,grid.at[posx,posy,recipe_done[0]].set(0).at[posx,posy,recipe_done[1]].set(0).at[posx,posy,recipe_done[2]].set(1),grid)
    # reward=reward+cost

    # drop  only if right recipe otherwise stay in inventory
    grid = grid.at[posx, posy, inventory].add(1)
    # inventory=0
    cost = -0.
    # test recipe
    recipe_done, reward = jax.lax.cond(grid[posx, posy, 1:].sum() == 2, test_recipes,
                                       lambda x, y: (jnp.zeros(3, jnp.int32), 0.), *(grid[posx, posy, :], recipes))
    grid = jnp.where(recipe_done[2] > 0,
                     grid.at[posx, posy, recipe_done[0]].set(0).at[posx, posy, recipe_done[1]].set(0).at[
                         posx, posy, recipe_done[2]].set(1), grid.at[posx, posy, inventory].set(0))
    inventory = jnp.where(recipe_done[2] > 0, 0, inventory)

    empty_inv = jnp.logical_and(grid[posx, posy, 1:].sum() == 0, inventory > 0)
    grid = jnp.where(empty_inv, grid.at[posx, posy, inventory].set(1), grid)
    inventory = jnp.where(empty_inv, 0, inventory)

    reward = reward + cost

    return grid, inventory, reward


def collect(grid, posx, posy, inventory, key):
    # inventory=jnp.where(grid[posx,posy,1:].sum()>0,jnp.argmax(grid[posx,posy,1:])+1,0)
    inventory = jnp.where(grid[posx, posy, 1:].sum() > 0,
                          jax.random.categorical(key, jnp.log(grid[posx, posy, 1:] / (grid[posx, posy, 1:].sum()))) + 1,
                          0)
    grid = jnp.where(inventory > 0, grid.at[posx, posy, inventory].add(-1), grid)
    return grid, inventory


class Gridworld(VectorizedTask):
    """gridworld task."""

    def __init__(self,
                 max_steps: int = 200,
                 test: bool = False, spawn_prob=0.005):
        self.max_steps = max_steps
        self.obs_shape = tuple([(AGENT_VIEW * 2 + 1) * (AGENT_VIEW * 2 + 1) * 3 + 3 + 1, ])
        self.act_shape = tuple([7, ])
        self.test = test

        def reset_fn(key):
            next_key, key = random.split(key)
            posx, posy = (1, 1)
            agent = AgentState(posx=posx, posy=posy, inventory=0)
            grid = get_init_state_fn(key)

            next_key, key = random.split(next_key)
            # permutation_recipe=jax.random.permutation(key,3)[:3]+1
            permutation_recipe = jnp.array([1, 2, 3])

            next_key, key = random.split(next_key)
            # grid_color=jnp.concatenate([jnp.array([[[[1,0,0]]]]),jax.random.choice(key,jnp.array([0.1,0.5,1.]),(1,1,5,3))],axis=2)
            grid_color = jnp.concatenate([jnp.array([[[[1, 0, 0]]]]), jax.random.uniform(key, (1, 1, 5, 3))], axis=2)
            # rand=jax.random.uniform(key)
            # permutation_recipe=jnp.where(rand>0.5,jnp.array([1,2,3]),jnp.array([1,3,2]))
            # permutation_recipe=jnp.where(rand<0.5,jnp.array([2,3,1]),permutation_recipe)
            return State(state=grid, obs=jnp.concatenate(
                [get_obs(state=grid, posx=posx, posy=posy, grid_color=grid_color), jnp.zeros(3), jnp.zeros(1)]),
                         last_action=jnp.zeros((7,)), reward=jnp.zeros((1,)), agent=agent,
                         steps=jnp.zeros((), dtype=int), grid_color=grid_color, permutation_recipe=permutation_recipe,
                         key=next_key, immo=0)

        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def rest_keep_recipe(key, recipes, grid_color, steps,reward):
            next_key, key = random.split(key)
            posx, posy = (1, 1)
            agent = AgentState(posx=posx, posy=posy, inventory=0)
            grid = get_init_state_fn(key)

            return State(state=grid, obs=jnp.concatenate(
                [get_obs(state=grid, posx=posx, posy=posy, grid_color=grid_color), jnp.zeros(3), jnp.zeros(1)]),
                         last_action=jnp.zeros((7,)), reward=jnp.ones((1,)) * reward, agent=agent,
                         steps=steps, permutation_recipe=recipes, grid_color=grid_color, key=next_key, immo=0)

        def step_fn(state, action):
            # spawn food
            grid = state.state
            reward = 0

            # move agent
            key, subkey = random.split(state.key)
            # maybe later make the agent to output the one hot categorical
            action = action* (state.immo <= 0)
            action = jax.nn.one_hot(action, 7)

            action_int = action.astype(jnp.int32)

            posx = state.agent.posx - action_int[1] + action_int[3]
            posy = state.agent.posy - action_int[2] + action_int[4]
            posx = jnp.clip(posx, 0, SIZE_GRID - 1)
            posy = jnp.clip(posy, 0, SIZE_GRID - 1)
            grid = grid.at[state.agent.posx, state.agent.posy, 0].set(0)
            grid = grid.at[posx, posy, 0].set(1)
            # collect or drop
            inventory = state.agent.inventory
            key, subkey = random.split(key)
            grid, inventory, reward = jax.lax.cond(jnp.logical_and(action[5] > 0, inventory > 0), drop,
                                                   (lambda a, b, c, d, e: (a, d, 0.)),
                                                   *(grid, posx, posy, inventory, state.permutation_recipe))
            grid, inventory = jax.lax.cond(jnp.logical_and(action[6] > 0, inventory == 0), collect,
                                           (lambda a, b, c, d, e: (a, d)), *(grid, posx, posy, inventory, subkey))

            steps = state.steps + 1
            r_done = jnp.logical_or(grid[:, :, -2].sum() > 0, steps > self.max_steps-1)
            done= steps > self.max_steps-1
            immo = jnp.where(reward < 0, 0, state.immo - 1)
            immo = jnp.clip(immo, 0, 1)

            # key, subkey = random.split(key)
            # rand=jax.random.uniform(subkey)
            # catastrophic=jnp.logical_and(steps>40,rand<1)
            # done=jnp.logical_or(done, catastrophic)
            # a=state.permutation_recipe[1]
            # b=state.permutation_recipe[2]
            # permutation_recipe=jnp.where(catastrophic,state.permutation_recipe.at[1].set(b).at[2].set(a), state.permutation_recipe)
            # steps = jnp.where(catastrophic, jnp.zeros((), jnp.int32), steps)

            cur_state = State(state=grid, obs=jnp.concatenate(
                [get_obs(state=grid, posx=posx, posy=posy, grid_color=state.grid_color),
                 (jnp.expand_dims(jax.nn.one_hot(inventory, 6)[1:], axis=-1) * state.grid_color[0, 0, 1:]).sum(0),
                 (immo / 25) * jnp.ones(1)]),
                              last_action=action, reward=jnp.ones((1,)) * reward,
                              agent=AgentState(posx=posx, posy=posy, inventory=inventory),
                              steps=steps, permutation_recipe=state.permutation_recipe, grid_color=state.grid_color,
                              key=key, immo=immo)

            # keep it in case we let agent several trials
            state = jax.lax.cond(
                r_done,
                lambda x: rest_keep_recipe(key, state.permutation_recipe, grid_color=state.grid_color, steps=steps,reward=reward),
                lambda x: x, cur_state)

            return state, reward, done

        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self,
             state: State,
             action: jnp.ndarray) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)


import gym
import numpy as np
import time


class Grid:
    def __init__(self,max_episode_steps=128):
        self.max_episode_steps=max_episode_steps
        self._env = Gridworld(max_steps=max_episode_steps)
        self.key=jax.random.PRNGKey(0)
        # Whether to make CartPole partial observable by masking out the velocity.


    @property
    def observation_space(self):
        self._observation_space = spaces.Box(
            low=-1.,
            high=1.0,
            shape=((2*AGENT_VIEW+1)*(2*AGENT_VIEW+1)*3+3+1+1+7,),
            dtype=np.float32)
        return self._observation_space

    @property
    def action_space(self):
        return spaces.Discrete(7)

    def reset(self):
        self._rewards = []
        self.key,key=jax.random.split(self.key)
        key=jax.random.split(key,16)
        state = self._env.reset(key)
        self.state=state
        obs=state.obs
        r=state.reward
        la=state.last_action
        return np.array(jnp.concatenate([obs,r,la],axis=1))

    def step(self, action):
        state, reward, done = self._env.step(self.state,action[:,0])
        self.state=state
        obs = state.obs
        reward=state.reward
        la = state.last_action
        obs=jnp.concatenate([obs,reward,la],axis=1)

        self._rewards.append(reward)
        if done[0]:
            self._rewards=np.array(self._rewards)

            info = [{"reward": sum(self._rewards[:,w]),
                    "length": self._rewards.shape[0]} for w in range(self._rewards.shape[1])]
            print(info)

        else:

            info = None
        return np.array(obs ), np.array(reward[:,0]) , np.array(done), info

    def render(self):
        print("not implemented yet")


    def close(self):
        print('not imp')