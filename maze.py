import pygame
import random
from collections import deque
import heapq
import matplotlib.pyplot as plt
import numpy as np
# Initialize Pygame

# Maze dimensions
width, height = 500, 500
maze_size = 20  # Adjust for a more complex maze
block_size = width // maze_size

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

NUM_TREASURES = 10

# Set up the display


# Player and treasures
#player_pos = [
    #random.randint(0, maze_size - 1),
    #random.randint(0, maze_size - 1)
#]
#treasures = []
#for _ in range(NUM_TREASURES):  # Number of treasures
  #while True:
    #treasure = [
        #random.randint(0, maze_size - 1),
        #random.randint(0, maze_size - 1)
    #]
    #if treasure not in treasures and treasure != player_pos:
      #treasures.append(treasure)
      #break


# Generating walls and obstacles dynamically
def generate_walls():
  walls = []
  for i in range(1, maze_size - 1):  # Avoid placing walls on the border
    for j in range(1, maze_size - 1):
      if [i,j] != player_pos \
         and [i,j] not in treasures \
         and random.choice([True, False, False]):
        walls.append([i, j])

  return walls


def generate_water(slope):
  water = []

  water_size = min(maze_size, maze_size) // 4

  start_x = random.randint(0, maze_size - water_size)
  start_y = random.randint(0, maze_size - water_size)

  # Fill the square with water
  for i in range(start_x, start_x + water_size):
    for j in range(start_y, start_y + water_size):
      water.append([i, j])

  return water


#slope = 0.5  # This is a placeholder; adjust your slope logic as needed
#walls = generate_walls()
#water = generate_water(slope)

#### Player movement
#
# Adicione aqui a lógica de seu jogador
#
# O objetivo de uma função de callback de movimento do jogador
# é determinar a próxima ação do jogador com base no estado atual
# do jogo. Ela deve ser capaz de interagir com o estado do jogo,
# como a posição atual do jogador e o layout do labirinto, para
# tomar decisões de movimento inteligentes ou aleatórias.
#
# - Pode acessar mais não modificar variáveis globais -
#
#  A posição de agua e de parede é dada por variáveis globais:
#
#  water = generate_water(slope)
#
#  walls = generate_walls()
#
#  Sugestão 1: Juntar todas campos em um único grafo (grid)
#  para percorrer de maneira única
#
#  Sugestão 2: Cria uma classe para representar o modelo de mundo e #  outra para encapsular a tomada de decisão
#


def combine_map():
  map_combined = []

  for i in range(maze_size):
    row = []
    for j in range(maze_size):
      # Adicione elementos de cada grade à grade 'map'
      if [i, j] in water:
        weight = 5
      else:
        weight = 1
      combined_element = {
          'treasure': [i, j] in treasures,
          'water': [i, j] in water,
          'wall': [i, j] in walls,
          'weight': weight
      }
      row.append(combined_element)

    # Adicione a linha combinada à grade 'map'
    map_combined.append(row)

  return map_combined

def dijkstra(map_combined, start, end):
  heap = [(0, start, [])]
  visited = set()

  while heap:
    cost, current, path = heapq.heappop(heap)

    if current in visited:
      continue

    visited.add(current)

    if current == end:
      return path

    row, col = current

    # Verifique os vizinhos (acima, abaixo, esquerda, direita)
    neighbors = [(row - 1, col), (row + 1, col), (row, col - 1),
                 (row, col + 1)]

    for neighbor in neighbors:
      n_row, n_col = neighbor
      if 0 <= n_row < maze_size and 0 <= n_col < maze_size and not map_combined[
          n_row][n_col]['wall']:
        new_cost = cost + map_combined[n_row][n_col].get(
            'weight', 1)  # Custo considerando o peso da agua
        heapq.heappush(heap, (new_cost, (n_row, n_col), path + [neighbor]))

  return None


def bfs(map_combined, start, end):
  queue = deque([(start, [])])
  visited = set()

  while queue:
    current, path = queue.popleft()
    if current == end:
      return path

    if current in visited:
      continue

    visited.add(current)

    row, col = current

    # Verifique os vizinhos (acima, abaixo, esquerda, direita)
    neighbors = [(row - 1, col), (row + 1, col), (row, col - 1),
                 (row, col + 1)]

    for neighbor in neighbors:
      n_row, n_col = neighbor
      if 0 <= n_row < maze_size and 0 <= n_col < maze_size and not map_combined[
          n_row][n_col]['wall']:
        queue.append(((n_row, n_col), path + [neighbor]))

  return None


def move_player(player_pos, treasures, map_combined, search, impossible_treasures, last_closest_treasure):
  #global player_pos, treasures, map_combined
  valid_treasures = [t for t in treasures if t not in impossible_treasures]
  if not valid_treasures:
    return 'GIVEUP'

  
  if last_closest_treasure not in treasures:
    closest_treasure = min(valid_treasures, key=lambda t: manhattan_distance(player_pos, t))
    last_closest_treasure = closest_treasure
  else:
    closest_treasure = last_closest_treasure  
  if search == 'd':
    path_to_treasure = dijkstra(map_combined, tuple(player_pos),
                                tuple(closest_treasure))
  if search == 'b':
    path_to_treasure = bfs(map_combined, tuple(player_pos),
                                tuple(closest_treasure))

  if path_to_treasure:
    next_step = path_to_treasure[0]

    if next_step[0] == player_pos[0] - 1:
      return 'UP'
    elif next_step[0] == player_pos[0] + 1:
      return 'DOWN'
    elif next_step[1] == player_pos[1] - 1:
      return 'LEFT'
    elif next_step[1] == player_pos[1] + 1:
      return 'RIGHT'
    
  else:
    impossible_treasures.append(closest_treasure)
    return "NONE"  



def manhattan_distance(pos1, pos2):
  return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def manual_move():
  events = pygame.event.get()
  #print(events)
  for event in events:
    #        print (event)
    if event.type == pygame.QUIT:
      return "GIVEUP"
    elif event.type == pygame.KEYDOWN:
      if event.key == pygame.K_w:
        return "UP"
      elif event.key == pygame.K_s:
        return "DOWN"
      elif event.key == pygame.K_a:
        return "LEFT"
      elif event.key == pygame.K_d:
        return "RIGHT"
      elif event.key == pygame.K_ESCAPE:
        return "GIVEUP"
  return "NONE"

def run_game(player_pos, treasures, map_combined, search):
  pygame.init()
  screen = pygame.display.set_mode((width, height))
  pygame.display.set_caption('Maze Treasure Hunt')

  # Load treasure image
  treasure_image = pygame.image.load('treasure.png')
  treasure_image = pygame.transform.scale(treasure_image,
                                        (block_size, block_size))
  # Game loop
  running = True
  score = 0
  steps = 0
  impossible_treasures = []
  last_closest_treasure = ()
  while running:
    direction = move_player(player_pos, treasures, map_combined, search, impossible_treasures, last_closest_treasure)

    score -= 1
  
    next_pos = player_pos
    if direction == 'UP':
      next_pos = (player_pos[0] - 1, player_pos[1])
    elif direction == 'DOWN':
      next_pos = (player_pos[0] + 1, player_pos[1])
    elif direction == 'LEFT':
      next_pos = (player_pos[0], player_pos[1] - 1)
    elif direction == 'RIGHT':
      next_pos = (player_pos[0], player_pos[1] + 1)
    elif direction == "NONE":
      score += 1
      steps -= 1
    else:
      print("Giving up")
      running = False
  
    #print(next_pos, maze_size)
    px, py = next_pos
    if [px, py] not in walls \
       and 0 <= next_pos[0] < maze_size \
       and 0 <= next_pos[1] < maze_size:
      player_pos = next_pos
    else:
      print("Invalid move!", next_pos)
      continue
  
    # Drawing
    screen.fill(BLACK)
    for row in range(maze_size):
      for col in range(maze_size):
  
        rect = pygame.Rect(col * block_size, row * block_size, block_size,
                           block_size)
        if [col, row] in walls:
          pygame.draw.rect(screen, BLACK, rect)
        elif [col, row] in water:
          pygame.draw.rect(screen, BLUE, rect)
        else:
          pygame.draw.rect(screen, WHITE, rect)
        if [col, row] == [px, py]:
          pygame.draw.rect(screen, RED, rect)
        elif [col, row] in treasures:
          pygame.draw.rect(screen, WHITE, rect)
          screen.blit(treasure_image, (col * block_size, row * block_size))
  
    if [px, py] in treasures:
      treasures.remove([px, py])
      print("Treasure found! Treasures left:", len(treasures))
  
    if [px, py] in water:
      score -= 5
      print("In water! Paying heavier price:", [px, py])
  
    pygame.display.flip()
    pygame.time.wait(100)  # Slow down the game a bit
    steps += 1
    if not treasures:
      running = False
    if steps >= 80:
      print(f"Maximum number of steps {steps}")
      running = False
  
  found_treasures = NUM_TREASURES - len(treasures)
  print(f"Found {found_treasures} treasures")
  final_score = (found_treasures * 500) + score
  print(f"Final score: {final_score}")
  pygame.quit()
  return final_score

scores_d = []
scores_b = []
for _ in range(15):
    player_pos = [
      random.randint(0, maze_size - 1),
      random.randint(0, maze_size - 1)
    ]
    treasures = []
    for _ in range(NUM_TREASURES):  # Number of treasures
      while True:
        treasure = [
            random.randint(0, maze_size - 1),
            random.randint(0, maze_size - 1)
        ]
        if treasure not in treasures and treasure != player_pos:
          treasures.append(treasure)
          break


    slope = 0.5  # This is a placeholder; adjust your slope logic as needed
    walls = generate_walls()
    water = generate_water(slope)
    map_combined = combine_map()
  
    final_score_bfs = run_game(player_pos, treasures, map_combined, 'b')
    scores_b.append(final_score_bfs)

    # Armazena o score da execução atual na lista de scores
for _ in range(15):
    player_pos = [
      random.randint(0, maze_size - 1),
      random.randint(0, maze_size - 1)
    ]
    treasures = []
    for _ in range(NUM_TREASURES):  # Number of treasures
      while True:
        treasure = [
            random.randint(0, maze_size - 1),
            random.randint(0, maze_size - 1)
        ]
        if treasure not in treasures and treasure != player_pos:
          treasures.append(treasure)
          break


    slope = 0.5  # This is a placeholder; adjust your slope logic as needed
    walls = generate_walls()
    water = generate_water(slope)
    map_combined = combine_map()
  
    final_score_dijkstra = run_game(player_pos, treasures, map_combined, 'd')
    scores_d.append(final_score_dijkstra) 

# Tabela

print("Scores de cada execução:")
print("Execução | Score BFS | Score Dijkstra")
for i, (score1, score2) in enumerate(zip(scores_b, scores_d), 1):
    print(f"{i:8d} | {score1:9d} | {score2:9d}")

# Cálculo da média
mean_score_b = np.mean(scores_b)
mean_score_d = np.mean(scores_d)
print(f"Média dos scores BFS: {mean_score_b:.2f}")
print(f"Média dos scores Dijkstra: {mean_score_d:.2f}")

# Gráfico de barras
bar_width = 0.35
x = np.arange(len(scores_b))

fig, ax = plt.subplots()
bar1 = ax.bar(x - bar_width/2, scores_b, bar_width, label='Score BFS')
bar2 = ax.bar(x + bar_width/2, scores_d, bar_width, label='Score Dijkstra')

ax.axhline(y=mean_score_b, color='g', linestyle='--', label=f'Média BFS: {mean_score_b:.2f}')
ax.axhline(y=mean_score_d, color='r', linestyle='--', label=f'Média Dijkstra: {mean_score_d:.2f}')

ax.set_xlabel('Execução')
ax.set_ylabel('Score')
ax.set_title('Desempenho do Jogo ao Longo das Execuções')
ax.set_xticks(x)
ax.legend()

# Adiciona rótulos de valor em cada barra
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(bar1)
autolabel(bar2)

plt.grid(True)
plt.show()