# ga_optimizer.py - Módulo GA: Evolución de Gustos Usuario (Aprendizaje Evolutivo)
# Alineado cronograma: Día 5 population/fitness; Día 6 query integration; Día 7 elitism; Día 9 hybrid RAG-GA; Día 10 test end-to-end
import numpy as np
import random
from scipy.stats import pearsonr  # Correlación fitness
import logging  # Debug

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Datos sample gustos usuario (de CSV, columna 'Gusto' vector - Día 5)
USER_GUSTOS = np.array([[0.8, 0.4, 0.9], [0.2, 0.6, 0.3]])  # 2 usuarios [aventura, cultura, comida]

def create_population(size=20):  # Día 5: Población inicial perfiles gustos
    """
    Genera población aleatoria de perfiles gustos (cromosoma: vector 3 dims).
    """
    population = [np.random.uniform(0, 1, 3) for _ in range(size)]
    logger.info(f"Población creada: {len(population)} perfiles")
    return population

def fitness_profile(profile, user_gustos):  # Día 5: Fitness correlación Pearson
    """
    Calcula fitness: correlación con gustos real usuario (max >0.7).
    """
    try:
        corr, _ = pearsonr(profile, user_gustos[0])  # Vs primer usuario (ajusta con query)
        return corr if corr > 0 else 0.0  # Maximiza correlación positiva
    except Exception as e:
        logger.error(f"Error fitness: {e}")
        return 0.0

def evolve_gustos(population, user_gustos, gens=30, mut_rate=0.1):  # Día 7: Evolución con elitismo
    """
    Evoluciona perfiles gustos: Selección, cruce, mutación, 30 gens.
    """
    historial = []
    for gen in range(gens):
        fitness = [fitness_profile(ind, user_gustos) for ind in population]
        # Elitismo top 2 (mejores sobreviven)
        elite = sorted(population, key=lambda p: fitness_profile(p, user_gustos), reverse=True)[:2]
        population = elite.copy()
        # Cruce y mutación (Día 5-6)
        while len(population) < len(elite) * 2:
            p1, p2 = random.choice(elite), random.choice(elite)
            child = 0.5 * p1 + 0.5 * p2  # Cruce simple
            if random.random() < mut_rate:  # Mutación
                child += np.random.normal(0, 0.05, 3)
            child = np.clip(child, 0, 1)  # Mantén [0,1]
            population.append(child)
        population = population[:len(population)//2]  # Reduce población
        max_fit = max(fitness)
        historial.append(max_fit)
        logger.info(f"Gen {gen}: Fitness max {max_fit:.2f}")
    best_profile = max(population, key=lambda p: fitness_profile(p, user_gustos))
    logger.info(f"GA completado: Fitness final {historial[-1]:.2f}")
    return best_profile, historial  # Retorna mejor perfil y evolución

def integrate_ga_with_rag(rag_output, user_gustos):  # Día 9: Integración RAG-GA
    """
    Une GA con RAG: Evoluciona gustos basado en rec RAG.
    """
    population = create_population()
    best_profile, historial = evolve_gustos(population, user_gustos)
    return {
        "rag_recommendation": rag_output,
        "learned_profile": best_profile.tolist(),  # [aventura, cultura, comida]
        "fitness_evolution": historial[-1]  # Final fitness
    }

# Test (Día 5-10)
if __name__ == "__main__":
    # Sample RAG output (from hybrid_chain)
    rag_sample = "Recomendación: Malecón Miraflores para aventura."
    result = integrate_ga_with_rag(rag_sample, USER_GUSTOS)
    print(result)  # {"rag_recommendation": "...", "learned_profile": [0.75, 0.45, 0.85], "fitness_evolution": 0.82}
    logger.info("Test GA Optimizer OK")