import retro

def game_loader(gameName: str, stageName: str):
    """
    loads the game and the specific stage from the retro emulator
    :param gameName: name of the retro game to load
    :param stageName: name of the stage to load
    :return: environment for game object
    """
    env = retro.make(gameName,stageName)

    return env


