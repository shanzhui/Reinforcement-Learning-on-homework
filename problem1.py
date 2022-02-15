import sys
sys.path.append( 'Lib' )
from Lib.RL import *

def plot_value(maze : Maze , value_map, desc='value plot' , fs = (20,20) ):
    plt.figure( figsize = fs )
    plt.imshow(value_map, cmap='hot', interpolation='nearest')
    for r in range( maze.r ):
        for c in range( maze.c ):
            p = np.array( (r,c) )
            if maze.check_equal( p , maze.goal , maze.wall ):
                continue
            s = '{:.2f}'.format (value_map[r,c])
            plt.text( x = c,y = r,s = f'value:{s}' , ha = 'center' , color = 'y' ,fontsize=24,
                     weight='bold' ).set_backgroundcolor('black')
    for g in maze.goal:
        r,c = g
        plt.text( x = c,y = r,s = f'goal' , ha = 'center' , color = 'g',fontsize=48 ).set_backgroundcolor('#965786')
    for w in maze.wall:
        r,c = w
        plt.text( x = c,y = r,s = f'wall' , ha = 'center' , color = 'black' ,fontsize=48).set_backgroundcolor('#965786')
    # plt.axis('off')
    plt.suptitle(desc , fontsize = 48)
    # plt.legend()
    plt.savefig( desc + '.jpg' )
    plt.show()
    
def plot_policy(maze : Maze , policy_map, value_map , actions, desc='policy plot' , fs = (20,20) ):
    plt.figure( figsize = fs )
    plt.imshow(value_map, cmap='hot', interpolation='nearest')
    for r in range( maze.r ):
        for c in range( maze.c ):
            p = np.array( (r,c) )
            if maze.check_equal( p , maze.goal , maze.wall ):
                continue
            s = policy_map[r,c]
            plt.text( x = c,y = r,s = f'{s}' , ha = 'center' , color = 'y' ,fontsize=48,
                     weight='bold' ).set_backgroundcolor('black')
    for g in maze.goal:
        r,c = g
        plt.text( x = c,y = r,s = f'goal' , ha = 'center' , color = 'g',fontsize=48 ).set_backgroundcolor('#965786')
    for w in maze.wall:
        r,c = w
        plt.text( x = c,y = r,s = f'wall' , ha = 'center' , color = 'black' ,fontsize=48).set_backgroundcolor('#965786')
    # plt.axis('off')
    plt.suptitle(desc , fontsize = 48)
    # plt.legend()
    plt.savefig( desc + '.jpg' )
    plt.show()
    


maze = Maze( 5,5,[ [0,0] , [0,4] , [4,0] , [4,4] ] , [ [1,1] , [1,3] , [3,1] , [3,3] ] )

policy = Policy()
policy.run( maze )
plot_value( maze , policy.value , desc = 'Value of Policy Iteration' )
plot_policy( maze , policy.policy ,  policy.value ,  actions , desc = 'Policy of Policy Iteration' )


value = Value()
value.run( maze )
plot_value( maze , value.value , desc = 'Policy of Value Iteration' )
plot_policy( maze , value.policy ,  policy.value ,  actions , desc = 'Value of Value Iteration' )