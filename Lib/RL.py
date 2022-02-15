import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import seaborn as sns
import copy
import json
import sys
import random

from random import randint, choice
from enum import Enum

actions = { '↑': [-1, 0], '↓': [1, 0], '←': [0, -1], '→': [0, 1]}

class MAP_ENTRY_TYPE(Enum):
    MAP_DOOR = -1,
    MAP_BLOCK = 1,
    MAP_EMPTY = 0,

class WALL_DIRECTION(Enum):
    WALL_LEFT = 0,
    WALL_UP = 1,
    WALL_RIGHT = 2,
    WALL_DOWN = 3,

class Map():
    def __init__(self, height, width):
        self.width = width
        self.height = height
        self.map = [[0 for x in range(self.width)] for y in range(self.height)]
    
    def resetMap(self, value):
        for y in range(self.height):
            for x in range(self.width):
                self.setMap(x, y, value)
    
    def setMap(self, x, y, value):
        if value == MAP_ENTRY_TYPE.MAP_EMPTY:
            self.map[y][x] = 0
        elif value == MAP_ENTRY_TYPE.MAP_BLOCK:
            self.map[y][x] = 1
    
    def isVisited(self, x, y):
        return self.map[y][x] != 1

    def showMap(self):
        for row in self.map:
            s = ''
            for entry in row:
                if entry == 0:
                    s += ' 0'
                elif entry == 1:
                    s += ' *'
                else:
                    s += ' X'
            print(s)

def checkAdjacentPos(map, x, y, width, height, checklist):
    directions = []
    if x > 0:
        if not map.isVisited(2*(x-1)+1, 2*y+1):
            directions.append(WALL_DIRECTION.WALL_LEFT)
                
    if y > 0:
        if not map.isVisited(2*x+1, 2*(y-1)+1):
            directions.append(WALL_DIRECTION.WALL_UP)

    if x < width -1:
        if not map.isVisited(2*(x+1)+1, 2*y+1):
            directions.append(WALL_DIRECTION.WALL_RIGHT)
        
    if y < height -1:
        if not map.isVisited(2*x+1, 2*(y+1)+1):
            directions.append(WALL_DIRECTION.WALL_DOWN)
        
    if len(directions):
        direction = choice(directions)
        #print("(%d, %d) => %s" % (x, y, str(direction)))
        if direction == WALL_DIRECTION.WALL_LEFT:
                map.setMap(2*(x-1)+1, 2*y+1, MAP_ENTRY_TYPE.MAP_EMPTY)
                map.setMap(2*x, 2*y+1, MAP_ENTRY_TYPE.MAP_EMPTY)
                checklist.append((x-1, y))
        elif direction == WALL_DIRECTION.WALL_UP:
                map.setMap(2*x+1, 2*(y-1)+1, MAP_ENTRY_TYPE.MAP_EMPTY)
                map.setMap(2*x+1, 2*y, MAP_ENTRY_TYPE.MAP_EMPTY)
                checklist.append((x, y-1))
        elif direction == WALL_DIRECTION.WALL_RIGHT:
                map.setMap(2*(x+1)+1, 2*y+1, MAP_ENTRY_TYPE.MAP_EMPTY)
                map.setMap(2*x+2, 2*y+1, MAP_ENTRY_TYPE.MAP_EMPTY)
                checklist.append((x+1, y))
        elif direction == WALL_DIRECTION.WALL_DOWN:
            map.setMap(2*x+1, 2*(y+1)+1, MAP_ENTRY_TYPE.MAP_EMPTY)
            map.setMap(2*x+1, 2*y+2, MAP_ENTRY_TYPE.MAP_EMPTY)
            checklist.append((x, y+1))
        return True
    else:
        return False
        
        
# random prim algorithm
def randomPrim(map, width, height):
    startX, startY = (randint(0, width-1), randint(0, height-1))
    map.setMap(2*startX+1, 2*startY+1, MAP_ENTRY_TYPE.MAP_DOOR)
    
    checklist = []
    checklist.append((startX, startY))
    while len(checklist):
        entry = choice(checklist)
        if not checkAdjacentPos(map, entry[0], entry[1], width, height, checklist):
            checklist.remove(entry)
            
def doRandomPrim(map):
    map.resetMap(MAP_ENTRY_TYPE.MAP_BLOCK)	
    randomPrim(map, (map.width-1)//2, (map.height-1)//2)

def run(WIDTH = 20, HEIGHT = 20):
    map = Map(WIDTH, HEIGHT)
    doRandomPrim(map)
    # map.showMap()
    return map

class Object:
    def __init__(self):
        pass
    def _check( self , cur : np.array , ls : list ):
        for p in ls:
            if np.array_equal( cur , p ):
                return True
        return False
    def _check_boundary( self , cur , maze ):
        if cur[0] < 0 or cur[0] >= maze.r or cur[1] < 0 or cur[1] >= maze.c:
            return False
        return True
    def check_equal( self , cur : np.array , goal : list , wall : list ):
        if self._check( cur , goal ) or self._check( cur , wall ) : 
            return True
        return False
    def initialize( self , shape : tuple ):
        self.policy = np.array( random.choices( list( actions.keys() ) , k = shape[0] * shape[1] ) ).reshape(shape).astype(object)
        self.value = np.zeros( shape )
    def run( self ):
        pass
    
class Maze(Object):
    def __init__(self , r : int = None , c : int  = None , goal : list = None , wall : list = None , MAX = 50 ):
        if r is not None:
            self.r = r
        else:
            self.r = np.random.randint(10,MAX)
        if c is not None:
            self.c = c
        else:
            self.c = np.random.randint(10,MAX)
        self.maze = [ (self.c) * [0]  for i in range( self.r ) ]
        self.goal = goal
        self.wall = wall
        if  self.goal is not None:
            self.goal_dic = { ( p[0] , p[1] ) : 1 for p in self.goal }
        else :
            self.goal = []
            self.goal_dic = {}
        if self.wall is not None:
            self.wall_dic = { ( p[0] , p[1] ) : 1 for p in self.wall }
        else :
            self.wall = []
            self.wall_dic = {}
        # self.show()
        self.run_maze = np.array( self.maze )
    def show( self ):
        maze = copy.deepcopy(self.maze)
        maze = [ [ str( c ) for c in r ] for r in maze]
        if self.goal is not None:
            for p in self.goal:
                maze[ p[0] ][ p[1] ] = 'O';
        if self.wall is not None:
            for p in self.wall:
                maze[ p[0] ][ p[1] ] = '*';
        print( np.array( maze ) )
        del maze
    def show_maze( self , title = '' ):
        data = np.zeros( np.array( self.maze ).shape )
        for r in range( self.r ):
            for c in range( self.c ):
                tp = tuple( ( r,c ) )
                if tp in self.reward_dict:
                    data[ r,c ] = self.reward_dict[ tp ]
        x = int( data.shape[0]/3 )
        y = int( data.shape[1]/3 )
        fig = plt.figure(figsize=(y,x))
        sns.heatmap(data)
        plt.title( f'The Maze And Reward {title}' )
        plt.savefig( title + '.jpg' )
        plt.show()
    def get_shape(self):
        return tuple( self.run_maze.shape )
    def change( self,x,y,v ):
        try:
            self.maze[x][y] = v
        except:
            print( 'oah!,the input is error !' , 'please check the coordinate and value!' )
    def generator( self , absolute_max_value = 10 ):
        self.maze = run( self.r , self.c ).map
        self.states = []
        self.reward_dict = dict()
        for i in range( self.r ):
            for j in range( self.c ):
                # print( i , j )
                if i != 0 and j != 0 and i != self.r-1 and j != self.c-1: 
                    self.reward_dict[ ( i,j )  ] = np.random.rand() * absolute_max_value * 2 - absolute_max_value
                    if self.maze[ i ][ j ] == 0:
                        self.states.append( (i,j) )
        duan = int( 0.1 * len( self.states ) )
        np.random.shuffle( self.states )
        for d in range(duan):
            self.reward_dict[ self.states[d] ] = 100
        self.start = self.states[-1]
        # self.show()
    def begin(self):
        return self.start
    def step( self , cur , action ):
        # print( cur , action )
        nex = np.add( np.array(cur) , np.array(action) )
        if tuple(nex) not in self.reward_dict:
            return cur , self.reward_dict[ tuple( cur ) ]
        else:
            return nex , self.reward_dict[ tuple( nex ) ]
class Policy(Object):
    def policy_func( self,nex,maze,num ):
        if self.check_equal( nex , maze.goal , maze.wall ) :
            return 0
        else:
            return 1.0/num
    def __init__( self ,episode = 10 ,  discount = 1 , iteration = 10 , 
                 virualize : bool = True , reward_func = lambda x , y : -1 ):
        self.episode = episode
        self.discount = discount
        self.iteration = iteration
        self.virualize = virualize
        self.policy_maze = None
        self.value_maze = None
        # self.policy_func = policy_func
        self.reward_func = reward_func
    def _policyevaludate( self , maze ):
        
        # for i in range( self.iteration ):
            value_old = self.value.copy()
            for r in range( maze.r ):
                for c in range(maze.c):
                    state = np.array( [ r , c ] )
                    if self.check_equal( state , [ np.array(p) for p in maze.goal ] , [ np.array(p) for p in maze.wall ] ) or not self._check_boundary(state , maze):
                        continue
                    
                    action = self.policy[ state[0] , state[1] ][0]
                    neighbour_states = [np.add(state, actions[command]) for command in actions]
                    num = len([neighbour for neighbour in neighbour_states if self._check_boundary( neighbour , maze ) and not self._check( 
                        state , [ np.array(p) for p in maze.wall ] ) ] ) 
                    neighbour_states = [neighbour for neighbour in neighbour_states if self._check_boundary( neighbour , maze ) ]
                    reward = self.reward_func( state , actions[action] )
                    expected_next_value = self.discount * np.sum([self.policy_func(neighbour, maze , num) *
                                                value_old[neighbour[0], neighbour[1]] for neighbour in
                                                neighbour_states])
                    self.value[state[0], state[1]] = reward + expected_next_value

    def _policyimprove( self , maze ):
        for r in range( maze.r ):
            for c in range( maze.c ):
                state = np.array( [ r , c ] )
                if self.check_equal( state , [ np.array(p) for p in maze.goal ] , [ np.array(p) for p in maze.wall ] ):
                        continue
                
                action_values = {}
                multi = {}
                for action in actions:
                    neighbour = np.add( state , np.array(actions[action]) )
                    if self._check( neighbour ,  [ np.array(p) for p in maze.wall ] ) or not self._check_boundary(neighbour , maze):
                        continue
                    reward = self.reward_func( neighbour , actions[action] )
                    expected_next_value = self.discount * self.value[neighbour[0], neighbour[1]]
                    val = reward + expected_next_value
                    action_values[action] = val
                    if val not in multi:
                        multi[val] = [action]
                    else:
                        multi[val].append( action )
                best_action = max(action_values, key=action_values.get)
                self.policy[state[0], state[1]] = ''.join( list( multi[ action_values[best_action] ] ) )
    def run( self , maze : Maze ):
        self.initialize( maze.get_shape() )
        
        for e in tqdm( range( self.episode ) ):
            self._policyevaludate( maze )
            self._policyimprove( maze )
            if self.virualize :
                print( f'The Episode in Policy Iteration is {e} : ----------------------------------------------------------------------' )
                print( ' Now value matrix is :  \n' , self.value )
                print( ' Now policy matrix is :  \n' , self.policy )
    

class Value(Object):
    def policy_func( self,nex,maze,num ):
        if self.check_equal( nex , maze.goal , maze.wall ) :
            return 0
        else:
            return 1.0/num
    def __init__ ( self , discount :int = 1 , iteration : int = 10 , reward_func = lambda x , y : -1 , virualize : bool = True ):
        self.discount = discount
        self.iteration = iteration
        self.reward_func = reward_func
        self.virualize = virualize

    def run( self , maze : Maze ):
        self.initialize( maze.get_shape() )
        for i in tqdm( range( self.iteration ) ):
            value_old = self.value.copy()
            for r in range( maze.r ):
                for c in range( maze.c ):
                    state = np.array( [ r , c ] )
                    if self.check_equal( state , [ np.array(p) for p in maze.goal ] , [ np.array(p) for p in maze.wall ] ) or not self._check_boundary(state , maze):
                        continue
                    action_values = {}
                    multi = {}
                    for action in actions:
                        neighbour = np.add( state , np.array(actions[action]) )
                        if self._check( neighbour ,  [ np.array(p) for p in maze.wall ] ) or not self._check_boundary(neighbour , maze):
                            continue
                        reward = self.reward_func( neighbour , actions[action] )
                        expected_next_value = self.policy_func(neighbour, maze , 1) * self.discount * self.value[neighbour[0], neighbour[1]]
                        val = reward + expected_next_value
                        action_values[action] = val
                        if val not in multi:
                            multi[val] = [action]
                        else:
                            multi[val].append( action )
                        # print( neighbour )
                    best_action = max(action_values, key=action_values.get)
                    self.policy[state[0], state[1]] = ''.join( list( multi[ action_values[best_action] ] ) )
                    self.value[state[0], state[1]] = action_values[best_action]
            if self.virualize :
                print( f'The Episode in Value Iteration is {i} : ----------------------------------------------------------------------' )
                print( ' Now value matrix is :  \n' , self.value )
                print( ' Now policy matrix is :  \n' , self.policy )
                        
class QLearning:

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1, method = 'random'):
        self.actions = actions  
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.display_name="Q-Learning"
        print("Using Q-Learning ...")
        if method == 'random':
            self.func = lambda self , x : np.random.uniform() >= self.epsilon
        else:
            self.func = lambda self, x : self.q_table.loc[ x , : ].sum() != 0
        self.method = method
    def choose_action(self, observation):
        self.check_state_exist(observation)

        if self.func( self , observation ):
            # Choose argmax action
            state_action_values = self.q_table.loc[observation, :]
            action = np.random.choice(state_action_values[state_action_values == np.max(state_action_values)].index) # handle multiple argmax with random
        else:
            # Choose random action
            action = np.random.choice(self.actions)

        return action


    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_target = r + self.gamma * self.q_table.loc[s_, :].max() # max state-action value
        old = self.q_table.loc[s_, :].max()
        self.q_table.loc[s, a] = self.q_table.loc[s, a] + self.lr * (q_target - self.q_table.loc[s, a])
        new = self.q_table.loc[s_, :].max()
        return s_, self.choose_action(str(s_)) , old , new

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
class Sarsa:

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1, method = 'random'):
        self.actions = actions  
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.display_name="Sarsa"
        print("Using Sarsa ...")
        if method == 'random':
            self.func = lambda self , x : np.random.uniform() >= self.epsilon
        else:
            self.func = lambda self, x : self.q_table.loc[ x , : ].sum() != 0
        self.method = method
    def choose_action(self, observation):
        self.check_state_exist(observation)
        if self.func( self , observation ):
            # Choose argmax action
            state_action_values = self.q_table.loc[observation, :]
            action = np.random.choice(state_action_values[state_action_values == np.max(state_action_values)].index) # handle multiple argmax with random
        else:
            action = np.random.choice(self.actions)

        return action


    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)

        a_ = self.choose_action(str(s_)) # argmax action
        old = self.q_table.loc[s_, a_]
        q_target = r + self.gamma * self.q_table.loc[s_, a_] # max state-action value
        new = self.q_table.loc[s_, :].max()

        self.q_table.loc[s, a] = self.q_table.loc[s, a] + self.lr * (q_target - self.q_table.loc[s, a])

        return s_, a_  ,old , new 


    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
            
class SarsaLambda:

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1, lambda_decay=0.5 , method = 'random'):
        self.actions = actions  
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.lambda_decay = lambda_decay
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.e_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.display_name="Sarsa(λ)"
        print("Using Sarsa(λ) ...")
        if method == 'random':
            self.func = lambda self , x : np.random.uniform() >= self.epsilon
        else:
            self.func = lambda self, x : self.q_table.loc[ x , : ].sum() != 0
        self.method = method
    def choose_action(self, observation):
        self.check_state_exist(observation)
        if self.func( self , observation ):
            # Choose argmax action
            state_action_values = self.q_table.loc[observation, :]
            action = np.random.choice(state_action_values[state_action_values == np.max(state_action_values)].index) # handle multiple argmax with random
        else:
            # Choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)

        a_ = self.choose_action(str(s_)) # argmax action
        old = self.q_table.loc[s_, a_]
        q_target = r + self.gamma * self.q_table.loc[s_, a_] # max state-action value
        error = q_target - self.q_table.loc[s, a]
        self.e_table.loc[s, a] += 1

        self.q_table += self.lr * error * self.e_table # update state-action value for all states and actions

        self.e_table *= self.gamma * self.lambda_decay # decay the eligibility trace for all states and actions
        new = self.q_table.loc[s_, :].max()
        return s_, a_ , old  ,new

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

        if state not in self.e_table.index:
            # append new state to q table
            self.e_table = self.e_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.e_table.columns,
                    name=state,
                )
            )
class Problem(Object):
    def __init__(self , episodes = 50 , eps = 1e-18 , iteration = 100 ):
        self.episodes = episodes
        self.eps = eps
        self.MAX = sys.float_info.max
        self.diff = 0
        self.global_reward = []
        self.iteration = iteration
    def run( self  , maze : Maze , agent ):
        self.method = agent.method
        self.global_reward = []
        self.diff = self.MAX
        self.action_list = {}
        t = 0
        # maze.show_maze()
        for i in tqdm(range( self.iteration )):
            state = maze.begin()
            print(f'In episode {i} with {agent.display_name} {agent.method} policy, the input is {state} ', end='')
            # RL choose action based on state
            action = agent.choose_action(str(tuple(state)))
            self.action_list[ i ] = [ (state , action ) ]
            sum_reward = 0
            for episode in range( self.episodes ):
                state_, reward = maze.step(state , np.array(actions[action]))
                sum_reward += reward
                state, action , old , new =  agent.learn(str(tuple(state)), action, reward, str(tuple(state_)))
                state = json.loads( state.replace('(' , '[').replace( ')' , ']' ) )
                self.action_list[ i ].append( (state , action) )
                print( f' update {state} from {old} to {new}  =>' , end = '' )
            self.global_reward.append( sum_reward )
            if len( self.global_reward ) == 1:
                self.diff = self.global_reward[0]
            else:
                self.diff = abs( self.global_reward[-1] - self.global_reward[-2] )
        
            t += 1
            print('\n')
        print('program over -- Algorithm {} completed'.format(agent.display_name))
        return self.global_reward