import neat
from game import Game  # a class that runs one playthrough

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        game = Game()              # your wrapper around main()
        fitness = 0
        
        # run until dino crashes
        while not game.game_over:
            inputs = game.get_state()        # e.g. [dist,next_h,vel_y, speed]
            output = net.activate(inputs)    # e.g. [jump_score, duck_score, nothing_score]
            
            # pick action
            action = output.index(max(output))
            if action == 0:
                game.do_jump()
            elif action == 1:
                game.do_duck()
            else:
                game.do_nothing()
            
            game.step()       # advance one frame, update obstacles, check collision
            fitness += game.score_delta()
        
        genome.fitness = fitness


if __name__ == "__main__":
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward.txt')
    
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    
    winner = p.run(eval_genomes, n=50)   # run for 50 generations
    print("Best genome:\n", winner)
