# game.py
# import the draw module

def play_game():
    print("Inside play_game")
    # Simulating a simple game with a score
    result = "Player 2 wins"
    score = {"Player 1": 5, "Player 2": 10}
    return result, score

def main():
    result = play_game()

# this means that if this script is executed, then 
# main() will be executed
if __name__ == '__main__':
    main()