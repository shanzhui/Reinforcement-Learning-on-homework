import sys
sys.path.append( 'Lib' )
from Lib.RL import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def augment(xold,yold,numsteps):
    xnew = []
    ynew = []
    for i in range(len(xold)-1):
        difX = xold[i+1]-xold[i]
        stepsX = difX/numsteps
        difY = yold[i+1]-yold[i]
        stepsY = difY/numsteps
        for s in range(numsteps):
            xnew = np.append(xnew,xold[i]+s*stepsX)
            ynew = np.append(ynew,yold[i]+s*stepsY)
    return xnew,ynew
def generate_animation1( p1,p2 , path = 'tmp.gif' , expand = 5 ):
    fig = plt.figure(figsize=(20,10))
    plt.xlim(0,p1.episodes+1) 
    plt.ylim( min(min(p1.global_reward) , min(p2.global_reward)) - 1 , max(max( p1.global_reward ) , max( p2.global_reward )) + 1 ) 
    plt.xlabel('Episode',fontsize=20) 
    plt.ylabel('Reward',fontsize=20)
    plt.title('Training',fontsize=20)
    
    y1 = p1.global_reward
    x = list(range(len(y1)))
    x , y1 = augment( x,y1,expand )
    
    y2 = p2.global_reward
    x = list(range(len(y2)))
    x , y2 = augment( x,y2,expand )
    def animate(i):
    # print(i)
        if i == 2:
            t = sns.lineplot(x=x[:i], y=y1[:i], color="blue" , label = 'random policy')
            t = sns.lineplot(x=x[:i], y=y2[:i], color="red" , ax = t  , linestyle='--' , label = 'greedy policy')
        else:
            t = sns.lineplot(x=x[:i], y=y1[:i], color="blue")
            t = sns.lineplot(x=x[:i], y=y2[:i], color="red" , ax = t  , linestyle='--')
        t.legend(fontsize = 'xx-large')
        t.tick_params(labelsize=20)
        plt.setp(t.lines,linewidth=5)
    
    ani = animation.FuncAnimation(fig, animate, frames=len(y1), repeat=True , interval=1,)
    
    ani.save(path)
    return fig

def generate_animation2( maze : Maze , p,path = 'tmp.gif'):
    data = np.zeros( np.array( maze.maze ).shape )
    for r in range( maze.r ):
        for c in range( maze.c ):
            tp = tuple( ( r,c ) )
            if tp in maze.reward_dict:
                data[ r,c ] = maze.reward_dict[ tp ]
    x = int( data.shape[0]/2 )
    y = int( data.shape[1]/2 )
    fig = plt.figure(figsize=(x,y))

    plt.title('Training',fontsize=20)
    
    def animate(i):
        point , action = p.action_list[p.iteration-1][i]
        tmp = maze.reward_dict[ ( point[0] , point[1] ) ]
        print( f'{i}th steps in the position {point} with action {action} ,  reward in the one block :  {tmp}' )
        plt.imshow(data + 10, cmap='hot', interpolation='nearest')
        plt.text( x = point[1],y = point[0],s = f'{action}' , ha = 'center' , color = 'g',fontsize=4 ).set_backgroundcolor('#965786')
    
    ani = animation.FuncAnimation(fig, animate, frames=len(p.action_list[p.iteration-1]), repeat=True , interval=1000,)
    
    ani.save(path)
    return fig
maze = Maze()
maze.generator()
maze.show_maze('heatmap')
print( "==================================================QLearning====================================================" )
p1 = Problem( 100 )
p1.run( maze  , QLearning( list(actions.keys()) ) )
p2 = Problem( 100 )
p2.run( maze  , QLearning( list(actions.keys()) , method = 'greedy' ) )
generate_animation1( p1,p2 , path = 'QLearning-train.gif' )
print( f"The Path of QLearning with {p1.method} policy" )
generate_animation2( maze , p1 , path = f'{p1.method}-QLearning-path.gif' )
print( f"The Path of QLearning with {p2.method} policy" )
generate_animation2( maze , p2 , path = f'{p2.method}-QLearning-path.gif' )
print( "==================================================Sarsa====================================================" )
p1 = Problem( 100 )
p1.run( maze  , Sarsa( list(actions.keys()) ) )
p2 = Problem( 100 )
p2.run( maze  , Sarsa( list(actions.keys()) , method = 'greedy' ) )
generate_animation1( p1,p2 , path = 'Sarsa-train.gif' )
print( f"The Path of Sarsa with {p1.method} policy" )
generate_animation2( maze , p1 , path = f'{p1.method}-Sarsa-path.gif' )
print( f"The Path of Sarsa with {p2.method} policy" )
generate_animation2( maze , p2 , path = f'{p2.method}-Sarsa-path.gif' )
print( "==================================================QLearning====================================================" )
p1 = Problem( 100 )
p1.run( maze  , SarsaLambda( list(actions.keys()) ) )
p2 = Problem( 100 )
p2.run( maze  , SarsaLambda( list(actions.keys()) , method = 'greedy' ) )
generate_animation1( p1,p2 , path = 'SarsaLambda-train.gif' )
print( f"The Path of SarsaLambda with {p1.method} policy" )
generate_animation2( maze , p1 , path = f'{p1.method}-SarsaLambda-path.gif' )
print( f"The Path of SarsaLambda with {p2.method} policy" )
generate_animation2( maze , p2 , path = f'{p2.method}-SarsaLambda-path.gif' )