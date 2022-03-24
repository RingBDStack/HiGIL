def claim_is_same(claim_a, claim_b):
    claim_a_len = len(_normalize_claim(claim_a))
    claim_b_len = len(_normalize_claim(claim_b))
    return max(claim_a_len, claim_b_len) - min(claim_a_len, claim_b_len) <= 10


def _normalize_claim(claim):
    return claim.replace('-LRB-', '(').replace('-RRB-', ')').replace(' ', '')