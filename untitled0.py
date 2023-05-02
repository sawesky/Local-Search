import numpy as np
import random
import matplotlib.pyplot as plt

#N velicina kvadrata
#~nebitno jednom se poziva
def gen_solution(n):
    
    arr = np.arange(1, n**2 + 1)
    np.random.shuffle(arr)
    sol = arr.reshape(n,n)
    return sol

#~O(N^2)
def obj_func(sol, n):
    
    obj_sum = n * (n**2 + 1) / 2
    sol_error = 0;
    for i in range(n):
        row_sum = sum(sol[i, :])
        sol_error += abs(row_sum - obj_sum)
        col_sum = sum(sol[:, i])
        sol_error += abs(col_sum - obj_sum)
    major_diag = sol.diagonal()
    sol_error += abs(sum(major_diag) - obj_sum)
    minor_diag = np.fliplr(sol).diagonal()
    sol_error += abs(sum(minor_diag) - obj_sum)
    
    return sol_error

#~O(N^2)
def gen_next_solutions(sol):
    
    next_solutions = []
    n = len(sol)
    #unutar cw
    for i in range(n - 1):
        for j in range(n - 1):
            new_sol = sol.copy()
            temp = [sol[i, j], sol[i, j + 1], sol[i + 1, j + 1], sol[i + 1, j]]
            new_sol[i, j + 1], new_sol[i + 1, j + 1], new_sol[i + 1, j], new_sol[i, j] = temp
            next_solutions.append(new_sol)
    #gore/dole cw
    for j in range(n - 1):
        new_sol = sol.copy()
        temp = [sol[n - 1, j], sol[n - 1, j + 1], sol[0, j + 1], sol[0 , j]]
        new_sol[n - 1, j + 1], new_sol[0, j + 1], new_sol[0 , j], new_sol[n - 1, j] = temp
        next_solutions.append(new_sol)
    #levo/desno cw
    for i in range(n - 1):
        new_sol = sol.copy()
        temp = [sol[i, n - 1], sol[i, 0], sol[i + 1, 0], sol[i + 1 , n - 1]]
        new_sol[i, 0], new_sol[i + 1, 0], new_sol[i + 1, n - 1], new_sol[i, n - 1] = temp
        next_solutions.append(new_sol)        
    
    return next_solutions
        
#~O(N^2*m)   
def random_search(n, num_iterations):
    
    best_sol = None
    best_error = float('inf')
    for i in range(num_iterations): #m
        sol = gen_solution(n)
        sol_error = obj_func(sol, n) #N^2
        if sol_error < best_error:
            best_sol = sol
            best_error = sol_error
        if sol_error == 0:
            break
    
    return best_sol, best_error
        
#~ objf->N^2, next_sols->neigh~10~N^2, num_iter~m => O(N^4*m)  
def greedy(n, num_iterations):
    
    best_sol = None
    best_error = float('inf')
    sol = gen_solution(n)
    best_errors = []
    for i in range(num_iterations): #m
        sol_error = obj_func(sol, n)
        if sol_error < best_error:
            best_sol = sol
            best_error = sol_error
        if best_error == 0:
            break
        next_sols = gen_next_solutions(sol)
        next_sol_errors = []
        for i in range(len(next_sols)): #N^2
            next_sol_errors.append(obj_func(next_sols[i], n)) #N^2
        next_best = next_sols[np.argmin(next_sol_errors)]
        # if obj_func(next_best, n) > obj_func(sol, n):
        #     break
        sol = next_best
        best_errors.append(best_error)
    
    return best_sol, best_error, best_errors


#~ objf->N^2 next_sols->N^2 num_iter~m => O(N^2*m)
def simulated_annealing(n, num_iterations, start_temp, cooling):
    
    best_sol = gen_solution(n)
    best_error = obj_func(best_sol, n)
    temperature = start_temp
    best_errors = []
    acc_probs = []
    curr_temp = []
    last_accepted_prob = 1
    for i in range(num_iterations): #m
        if temperature <= 0.01:
            break
        next_sols = gen_next_solutions(best_sol) #N^2
        next_sol = next_sols[random.randint(0, len(next_sols) - 1)]
        next_sol_error = obj_func(next_sol, n)
        if next_sol_error < best_error:
            best_sol = next_sol
            best_error = next_sol_error
            if best_error == 0:
                break
        else:
            delta = next_sol_error - best_error
            accept_prob = np.exp(-delta/temperature)
            last_accepted_prob = accept_prob
            if random.random() < accept_prob:
                best_sol = next_sol
                best_error = next_sol_error
        if i % 5 == 0:
            temperature *= cooling
        best_errors.append(best_error)
        acc_probs.append(last_accepted_prob)
        curr_temp.append(temperature)
        
    return best_sol, best_error, best_errors, curr_temp, acc_probs

#~O(nlogn + N^4) za N = 10, N^4 = 10000 -> 100log100 << 10000 => O(N^4)
def selection(sols, n, beam):
    
    errors = []
    for sol in sols: #N^2
        errors.append(obj_func(sol, n)) #N^2
    min_error_indices = np.argsort(errors)[:beam] #N^2logN^2
    min_errors_sols = []
    for err in min_error_indices:
        min_errors_sols.append(sols[err])
    
    return min_errors_sols

#~ objf->N^2, next_sols->N^2, sel->N^4, num_iter~m => O(beam*N^4*m) 
def beam_search(n, num_iterations, beam):
    
    best_sol = gen_solution(n)
    best_error = obj_func(best_sol, n)
    next_sols = gen_next_solutions(best_sol) #N^2
    best_errors = []
    avg_errors = []
    curr_errors = []
    for i in range(num_iterations): #m
        selected_sols = selection(next_sols, n, beam) #N^4 prvi call, beam*N^4 sledeci
        for selected_sol in selected_sols: #beam
            error = obj_func(selected_sol, n) #N^2
            curr_errors.append(error)
            if error < best_error:
                best_sol = selected_sol
                best_error = error
            if error == 0:
                break
        if best_error == 0:
            break
        next_sols = []
        for selected_sol in selected_sols:
            next_sols.extend(gen_next_solutions(selected_sol)) #beam*N^2
        # avg_error = np.mean([obj_func(sol, n) for sol in next_sols])
        avg_error = np.mean(curr_errors)
        avg_errors.append(avg_error)
        best_errors.append(best_error)
        
    return best_sol, best_error, avg_errors, best_errors
                
#~ O(1)
def mutate(sol, n):
    
    i = random.randint(0, n - 1)
    j = random.randint(0, n - 1)
    temp = sol[i, j]
    sol[i, j] = sol[j, i]
    sol[j, i] = temp
    
    return sol

#~ O(N^2)
def crossover(parent1, parent2, n):
    
    child1 = np.zeros(n**2)
    child2 = np.zeros(n**2)
    set1 = set(i + 1 for i in range(n**2))
    set2 = set(i + 1 for i in range(n**2))
    parent1 = parent1.flatten()
    parent2 = parent2.flatten()
    # imam secenje na pola roditelja 1
    for i in range(n**2//2):
        child1[i] = parent1[i]
        set1.remove(parent1[i])
    for i in range(n**2//2, n**2):
        if parent2[i] in set1:
            child1[i] = parent2[i]
            set1.remove(parent2[i])
    for i in range(n**2//2, n**2):
        if child1[i] == 0:
            child1[i] = set1.pop()
    #isto na pola roditelja2
    for i in range(n**2//2):
        child2[i] = parent2[i]
        set2.remove(parent2[i])
    for i in range(n**2//2, n**2):
        if parent1[i] in set2:
            child2[i] = parent1[i]
            set2.remove(parent1[i])
    for i in range(n**2//2, n**2):
        if child2[i] == 0:
            child2[i] = set2.pop()    
    child1 = child1.reshape(n, n)
    child2 = child2.reshape(n, n)
    
    return child1, child2

#~ sel->n*N^2 objf->N^2 crossover->N^2*n num_iter~m => O(N^2*n*m)  // u odnosu na snop N^2*n = N^4*beam => beam*N^2 = n 
def genetic(n, num_iterations, population_size, elite_num):
    
    best_sol = None
    best_error = float('inf')
    population = []
    avg_errors = []
    best_errors = []
    for i in range(population_size): #n
        population.append(gen_solution(n))
    for i in range(num_iterations): #m
        avg_error = np.mean([obj_func(sol, n) for sol in population])
        elite = selection(population, n, elite_num) #n*N^2
        if obj_func(elite[0], n) < best_error:
            best_sol = elite[0]
            best_error = obj_func(elite[0], n)
        if best_error == 0:
            break
        errors = []
        for sol in population: #n
            errors.append(obj_func(sol, n)) #N^2
        weights = []
        for error in errors:
            weights.append(1.0/error)
        parents = random.choices(population, weights, k = population_size - elite_num)
        offspring = []
        for j in range(0, population_size - elite_num, 2): #n
            child1, child2 = crossover(parents[j], parents[j + 1], n) #N^2
            offspring.append(child1)
            offspring.append(child2)
        for mutable in offspring:
            if random.random() < 0.05:
                mutable = mutate(mutable, n)
        all_population = selection(offspring + parents, n, population_size) #2n*N^2 v 2nlog2n => 2n*N^2
        population = all_population + elite
        avg_errors.append(avg_error)
        best_errors.append(best_error)
    
    return best_sol, best_error, avg_errors, best_errors

#%%

# proba = gen_solution(3)
# k = obj_func(proba, 3)
# s = gen_next_solutions(proba)

# objer = []
# error_sim = []
# error_greedy = []
# for x in s:
#     objer.append(obj_func(x, 3)) 

# # crossover(gen_solution(3), gen_solution(3), 3)
# # best_sol_gen, best_error_gen = genetic(10, 100, 50, 10)
# # best_sol_ran, best_error_ran = random_search(10, 5000)
# # best_sol_greedy, best_error_greedy = greedy(10, 5000)
# # best_sol_sim, best_error_sim = simulated_annealing(10, 5000, 100, 0.95)    
# # best_sol_beam, best_error_beam = beam_search(10, 1000, 5)
# for j in range(40):
#     best_sol_sim, best_error_sim = simulated_annealing(10, 100, 100, 0.75)
#     error_sim.append(best_error_sim)
# for j in range(4):
#     best_sol_greedy, best_error_greedy, ret_errors = greedy(10, 10)
#     error_greedy.append(best_error_greedy)
#%% 100 iteracija

error_ran =[]
error_greedy = []
error_sim = []
error_beam = []
error_gen = []

for i in range(100):        
    best_sol_ran, best_error_ran = random_search(10, 4000)
    for j in range(4):
        best_sol_greedy, best_error_greedy, ret_errors = greedy(10, 10)
        error_greedy.append(best_error_greedy)
    for j in range(40):
        best_sol_sim, best_error_sim, sim_err, sim_temp, sim_prob  = simulated_annealing(10, 100, 150, 0.75)
        error_sim.append(best_error_sim)
    best_sol_beam, best_error_beam, avg_error_beam, curr_error_beam = beam_search(10, 10, 4)
    best_sol_gen, best_error_gen, avg_error_gen, curr_error_gen = genetic(10, 40, 100, 10)
    error_ran.append(best_error_ran)
    error_beam.append(best_error_beam)
    error_gen.append(best_error_gen)

#%% mean/std

mean_ran = np.mean(error_ran)
mean_greedy = np.mean(error_greedy)
mean_sim = np.mean(error_sim)
mean_beam = np.mean(error_beam)
mean_gen = np.mean(error_gen)

std_ran = np.std(error_ran)
std_greedy = np.std(error_greedy)
std_sim = np.std(error_sim)
std_beam = np.std(error_beam)
std_gen = np.std(error_gen)

#%% greedy

greedy_full = []
greedy_full_mean = []
greedy_full_std = []
for i in range(4):
    best_sol_greedy, best_error_greedy, errors_greedy = greedy(10, 10)
    greedy_full.append(errors_greedy)

greedy_full_mean = np.mean(greedy_full, axis = 0)
greedy_full_std = np.std(greedy_full, axis = 0)

fig, ax = plt.subplots()
ax.errorbar(range(len(greedy_full_mean)), greedy_full_mean, yerr=greedy_full_std, fmt='-o')
fig.suptitle('Greedy pretraga')
#%% sim

sim_temp = []
sim_probs = []
sim_full = []
sim_full_mean = []
sim_full_std = []
sim_probs_mean = []
sim_probs_std = []

for j in range(40):
    best_sol_sim, best_error_sim, sim_err, sim_temp, sim_prob  = simulated_annealing(10, 100, 150, 0.75)
    sim_full.append(sim_err)
    sim_probs.append(sim_prob)

sim_full_mean = np.mean(sim_full, axis = 0)
sim_full_std = np.std(sim_full, axis = 0)    
sim_probs_mean = np.mean(sim_probs, axis = 0)
sim_probs_std = np.std(sim_probs, axis = 0)

fig, ax = plt.subplots(3, 1, figsize=(8, 10))
ax[0].plot(range(len(sim_temp)), sim_temp)
ax[0].set_title('Temperatura')
ax[1].errorbar(range(len(sim_probs_mean)), sim_probs_mean, yerr=sim_probs_std, fmt = '-o')
ax[1].set_title('Verovatnoce prihvatanja')
ax[2].errorbar(range(len(sim_full_mean)), sim_full_mean, yerr=sim_full_std, fmt = '-o')
ax[2].set_title('Trenutna resenja')
fig.suptitle('Simulirano kaljenje', y=0.935)

#%% beam

best_sol_beam, best_error_beam, avg_error_beam, curr_error_beam = beam_search(10, 10, 4)

fig, ax = plt.subplots()
avg_plot, = ax.plot(range(len(avg_error_beam)), avg_error_beam, color = 'b')
curr_plot, = ax.plot(range(len(curr_error_beam)), curr_error_beam, color = 'r')
ax.set_title('Pretraga po snopu')
ax.legend([avg_plot, curr_plot], ['avg', 'best'])


#%% genetic

best_sol_gen, best_error_gen, avg_error_gen, curr_error_gen = genetic(10, 40, 100, 10)

fig, ax = plt.subplots()
avg_plot, = ax.plot(range(len(avg_error_gen)), avg_error_gen, color = 'b')
curr_plot, = ax.plot(range(len(curr_error_gen)), curr_error_gen, color = 'r')
ax.set_title('Genetski algoritam')
ax.legend([avg_plot, curr_plot], ['avg', 'best'])












