"""
LEAK INJECTION — NOT IMPLEMENTED IN THIS PASS.

When added, this module should expose:
  - inject_leak(node_id: str, rate_lpm: float) -> None
  - clear_leak(node_id: str) -> None
  - get_active_leaks() -> dict[str, float]

It must be called from simulation/aggregation.py AFTER the normal bottom-up
sum and BEFORE noise is applied, subtracting the leak rate from the affected
node and propagating the deficit upward through parent junctions per the
balance equations in section 2 of BACKEND_IMPLEMENTATION_PLAN.md.

The /datasink endpoints must then also start populating `leak`, `leak_rate`,
and `leak_zone` fields (currently hardcoded to 0/null) so classifier training
data includes labeled leak examples.
"""
