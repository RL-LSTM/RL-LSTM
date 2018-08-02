import numpy as np
import itertools
import scipy.misc
import matplotlib.pyplot as plt


def main():
    env = gameEnv(size=5,startDelay=2)
    # game main loop
    isGameOver = False
    while not isGameOver:
        while 1:
            action = key2action(str(input("next action:")))
            if action < 0 or action > 4:
                print("Invaild Action!")
                continue
            else:
                break
        state, reward, isGameOver,info = env.step(float(action))
        env.render()
        # plt.imshow(state, interpolation="nearest")
        # plt.title("Score: {0}, Reward: {1}, GameOver: {2}".format(float(env.getScore()),float(reward),isGameOver))
        # plt.draw()
        # plt.show(block=False)
    if env.getScore() == 1:
        input("Win!!!")
    else:
        input("Lose :-(")

def key2action(key):
    # 0 - up, 1 - down, 2 - left, 3 - right
    # print("key %s"%(key))
    if key == 'w':
        return 0
    if key == 's':
        return 1
    if key == 'a':
        return 2
    if key == 'd':
        return 3
    else:
        return -1

def colorByName(colorName):
    return {
        # colorName: [red,green,blue]
        'purple': [[0.5,0,1]],
        'red': [1,0,0],
        'green': [0,1,0],
        'blue': [0,0,1],
        'yellow': [1,1,0],
        'white': [1,1,1]
    }[colorName]

def colorByNum(colorNum):
    return {
        # colorName: [red,green,blue]
        0: colorByName('purple'),
        1: colorByName('red'),
        2: colorByName('green'),
        3: colorByName('blue'),
        4: colorByName('yellow'),
        5: colorByName('white')
    }[colorNum]

class gameOb():
    def __init__(self, coordinates, size, intensity, channel, reward, name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name

class gameEnv():
    def __init__(self, partial=0, size=10, fruitNum=0, holeNum=0, startDelay=4):
        class ActionSpace():
            def __init__(self):
                self.n = 4
        self.action_space = ActionSpace()
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.startDelayConst = startDelay
        self.partial = partial
        self.fruitNum = fruitNum
        self.holeNum = holeNum
        self.startItems = []
        self.objects = []
        self.done = False
        self.reward = 0
        self.penalty = 0
        self.score = 0
        self.numOfSteps = 0
        self.MaxNumOfStepsPerGame = self.sizeX*self.sizeY*10
        self.startDelay = self.startDelayConst
        self.objects = []
        #prepare first time for play
        self.reset()

    def seed(self,seedNum):
         np.random.seed(seedNum)

    def isPositionTaken(self,x,y,objects=None):
        if objects == None:
            objects = self.objects
        for objectA in objects:
            if (x, y) == (objectA.x,objectA.y):
                return True
        return False

    def createFruit(self,pos=None):
        if pos == None:
            pos = self.newPosition()
        if not self.isPositionTaken(pos[0],pos[1]):
            return gameOb(pos, 1, colorByName('green'), [0, 1, 2], 1, 'fruit')
        else:
            return None

    def createHole(self,pos=None):
        if pos == None:
            pos = self.newPosition()
        if not self.isPositionTaken(pos[0],pos[1]):
            return gameOb(pos, 1, colorByName('red'), [0, 1, 2], -1, 'hole')
        else:
            return None

    def createHero(self,pos=None):
        # WATCH OUT that hero have different color than items
        if pos == None:
            pos = self.newPosition()
        if not self.isPositionTaken(pos[0],pos[1]):
            return gameOb(pos, 1, colorByName('white'), [0, 1, 2], None, 'hero')
        else:
            return None

    def createItem(self,pos=None,color=colorByName('red')):
        if pos == None:
            pos = self.newPosition()
        if not self.isPositionTaken(pos[0],pos[1]):
            return gameOb(pos, 1, color, [0, 1, 2], 0, 'item')
        else:
            return None

    def reset(self):
        plt.close('all')
        self.objects = []
        # create items
        self.startItems = np.random.choice(self.startDelayConst, self.startDelayConst, replace=False)
        # # create bugs
        # for idx in range(self.fruitNum):
        #     self.objects.append(self.createFruit())
        # # create holes
        # for idx in range(self.holeNum):
        #     self.objects.append(self.createHole())
        # first init state
        self.done = False
        self.reward = 0
        self.penalty = 0
        self.score = 0
        self.startDelay = self.startDelayConst
        self.isSequenceFollowed = True
        state = self.renderEnv()
        self.state = state
        self.numOfSteps = 0
        return state

    def startPlay(self):
        # create hero
        self.objects = []
        self.objects.append(self.createHero())
        for colorNum in self.startItems:
            itemColor = colorByNum(colorNum)
            item = self.createItem(color=itemColor)
            self.objects.append(item)


    def moveChar(self, direction):
        # 0 - up, 1 - down, 2 - left, 3 - right
        hero = self.objects[0]
        heroX = hero.x
        heroY = hero.y
        penalize = 0.
        if direction == 0 and hero.y >= 1:
            hero.y -= 1
        if direction == 1 and hero.y <= self.sizeY - 2:
            hero.y += 1
        if direction == 2 and hero.x >= 1:
            hero.x -= 1
        if direction == 3 and hero.x <= self.sizeX - 2:
            hero.x += 1
        if hero.x == heroX and hero.y == heroY:
            penalize = 0.0
        self.objects[0] = hero
        # check if follow sequence - item[1] is always the next in the sequence
        isOnCorrectNextItem = ([hero.x,hero.y] == [self.objects[1].x,self.objects[1].y])
        isNotOnAnyItem = not self.isPositionTaken(hero.x,hero.y,self.objects[1:])
        self.isSequenceFollowed = self.isSequenceFollowed and (isNotOnAnyItem or isOnCorrectNextItem)
        return penalize

    def newPosition(self):
        iterables = [range(self.sizeX), range(self.sizeY)]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        currentPositions = []
        for objectA in self.objects:
            if (objectA.x, objectA.y) not in currentPositions:
                currentPositions.append((objectA.x, objectA.y))
        for pos in currentPositions:
            points.remove(pos)
        location = np.random.choice(range(len(points)), replace=False)
        return points[location]

    def checkGoal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        for other in others:
            if hero.x == other.x and hero.y == other.y:
                self.objects.remove(other)
                # if other.reward == 1:
                #     self.objects.append(self.createFruit())
                # else:
                #     self.objects.append(self.createHole())
                # check if game ended
                if len(self.objects) == 1:
                    return int(self.isSequenceFollowed),True
                else:
                    return 0,False
        return 0,False


    def renderEnv(self):
        # a = np.zeros([self.sizeY,self.sizeX,3])
        a = np.ones([self.sizeY + 2, self.sizeX + 2, 3])
        a[1:-1, 1:-1, :] = 0
        hero = None
        for item in self.objects:
            a[item.y + 1:item.y + item.size + 1, item.x + 1:item.x + item.size + 1, item.channel] = item.intensity
            if item.name == 'hero':
                hero = item
        if self.partial > 0:
            a = a[hero.y:hero.y + self.partial, hero.x:hero.x + self.partial, :]
        b = scipy.misc.imresize(a[:, :, 0], [84, 84, 1], interp='nearest')
        c = scipy.misc.imresize(a[:, :, 1], [84, 84, 1], interp='nearest')
        d = scipy.misc.imresize(a[:, :, 2], [84, 84, 1], interp='nearest')
        a = np.stack([b, c, d], axis=2)
        return a

    def render(self):
        # pass
        state = self.renderEnv()
        # show curent step
        plt.imshow(state, interpolation="nearest")
        plt.title("Score: {0}, Reward: {1}, GameOver: {2}".format(float(self.getScore()), float(self.reward), self.done))
        plt.draw()
        plt.show(block=False)
        # plt.hold(True)

    def step(self, action):
        # game pre-start
        if self.startDelay > 0:
            self.showNextItem()
            self.startDelay -= 1
            return self.renderEnv(),0,False,None
        # first game frame
        if self.startDelay == 0:
            self.startDelay -= 1
            self.startPlay()
            # return self.reset(),0,False,None
            return self.renderEnv(),0,False,None
        # rest of the game
        self.penalty = self.moveChar(action)
        self.reward, self.done = self.checkGoal()
        self.score += (self.reward+self.penalty)
        state = self.renderEnv()
        # check if too many step for game
        self.numOfSteps += 1
        if self.numOfSteps >= self.MaxNumOfStepsPerGame:
            return state, (self.reward + self.penalty), True, None
        else:
            return state, (self.reward + self.penalty), self.done, None


    def getScore(self):
        return self.score

    def showNextItem(self):
        # delete last item
        self.objects.clear()
        # add new item - select color by frame number (startDelay)
        itemColor = colorByNum(self.startItems[-self.startDelay])
        item = self.createItem([int(self.sizeX/2),int(self.sizeY/2)],itemColor)
        self.objects.append(item)


if __name__ == "__main__":
    main()
