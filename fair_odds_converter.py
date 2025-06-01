from scipy.stats import norm

def prob_over(line, mu, sigma):
    return 1 - norm.cdf(line, loc=mu, scale=sigma)

def prob_to_american(prob):
    if prob == 0:
        return None
    if prob > 0.5:
        return -round(prob / (1 - prob) * 100)
    else:
        return round((1 - prob) / prob * 100)

def fair_odds(prob):
    return 1 / prob if prob > 0 else None
