"""Interface commune des agents (§6.1) : un nom et une méthode `act`.

Les agents sont des programmes réactifs à rôle unique, déterministes, et pas des LLM :
chacun reçoit un message typé, applique sa règle et renvoie un message typé. Chaque
message est ensuite écrit dans le registre par l'orchestrateur.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

In = TypeVar("In", contravariant=True)
Out = TypeVar("Out", covariant=True)


class Agent(Protocol[In, Out]):
    @property
    def name(self) -> str: ...

    def act(self, message: In) -> Out: ...
