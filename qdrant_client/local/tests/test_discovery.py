import pytest

from qdrant_client import QdrantClient, models


@pytest.mark.parametrize("vector_kind", ["dense", "sparse", "multivector"])
@pytest.mark.parametrize("context_size", [1, 2])
def test_discovery_on_empty_collection(vector_kind: str, context_size: int) -> None:
    client = QdrantClient(":memory:")
    try:
        if vector_kind == "sparse":
            client.create_collection(
                "discovery",
                vectors_config={},
                sparse_vectors_config={"v": models.SparseVectorParams()},
            )
            positive = models.SparseVector(indices=[0], values=[1.0])
            negative = models.SparseVector(indices=[0], values=[-1.0])
            neutral = models.SparseVector(indices=[1], values=[1.0])
        else:
            multivector_config = (
                models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM)
                if vector_kind == "multivector"
                else None
            )
            client.create_collection(
                "discovery",
                vectors_config={
                    "v": models.VectorParams(
                        size=2,
                        distance=models.Distance.DOT,
                        multivector_config=multivector_config,
                    )
                },
            )
            positive = [[1.0, 0.0]] if vector_kind == "multivector" else [1.0, 0.0]
            negative = [[-1.0, 0.0]] if vector_kind == "multivector" else [-1.0, 0.0]
            neutral = [[0.0, 1.0]] if vector_kind == "multivector" else [0.0, 1.0]

        query = models.DiscoverQuery(
            discover=models.DiscoverInput(
                target=positive,
                context=[models.ContextPair(positive=positive, negative=negative)] * context_size,
            )
        )
        assert client.query_points("discovery", query=query, using="v").points == []

        # Check all three possible context ranks after the collection is populated.
        client.upsert(
            "discovery",
            [
                models.PointStruct(id=1, vector={"v": positive}),
                models.PointStruct(id=2, vector={"v": neutral}),
                models.PointStruct(id=3, vector={"v": negative}),
            ],
        )
        points = client.query_points("discovery", query=query, using="v").points
        assert [point.id for point in points] == [1, 2, 3]
        assert [point.score for point in points] == pytest.approx(
            [context_size + 0.75, 0.5, -context_size + 0.25]
        )
    finally:
        client.close()
