package semanticvg_cj.storage

import std.collection.*
import std.unittest.*
import std.unittest.testmacro.*

@Test
public class MixedScoringTest {
    @TestCase
    public func clampsBetaAndCombinesDistances(): Unit {
        @Assert(clampUnit(-0.5), 0.0)
        @Assert(clampUnit(1.5), 1.0)
        @Assert(mixedDistance(10.0, Some(2.0), 0.25), 8.0)
        @Assert(mixedDistance(10.0, None, 0.25), 10.0)
    }

    @TestCase
    public func extractsStorageBaseKey(): Unit {
        @Assert(storageBaseKey("bird-0001"), "bird")
        @Assert(storageBaseKey("bird-name"), "bird-name")
        @Assert(storageBaseKey("bird"), "bird")
    }
}

@Test
public class VectorMathTest {
    @TestCase
    public func computesNormalizedCosineDistance(): Unit {
        @Assert(dotProduct([1.0, 2.0], [3.0, 4.0]), 11.0)
        @Assert(cosineDistanceNormalized([1.0, 0.0], [1.0, 0.0]), 0.0)
        @Assert(cosineDistanceNormalized([1.0, 0.0], [0.0, 1.0]), 1.0)
    }

    @TestCase
    public func normalizesNonZeroVector(): Unit {
        let normalized = normalizedVector([3.0, 4.0])
        @Assert(normalized[0] > 0.599999999999 && normalized[0] < 0.600000000001, true)
        @Assert(normalized[1] > 0.799999999999 && normalized[1] < 0.800000000001, true)
    }
}

@Test
public class VectorGroupCrudTest {
    @TestCase
    public func keepsVectorsAndDescriptionsAligned(): Unit {
        let descriptions = Array<Any>(2, { i => "item-${i}" })
        let group = VectorGroup(
            "fixture",
            [1.0, 0.0],
            "fixture",
            [[1.0, 0.0], [0.0, 1.0]],
            descriptions
        )

        @Assert(group.size(), 2)
        @Assert(group.append([0.5, 0.5], "inserted"), 2)
        @Assert(group.updateAt(2, [0.25, 0.75]), true)
        @Assert(group.vectors[2][0], 0.25)
        @Assert(group.removeAt(1), true)
        @Assert(group.size(), 2)
        @Assert(group.descriptions.size, group.vectors.size)
        @Assert(group.updateAt(99, [1.0, 0.0]), false)
        @Assert(group.removeAt(99), false)
    }

    @TestCase
    public func rejectsInconsistentInitialDimensions(): Unit {
        var rejected = false
        try {
            let _ = VectorGroup(
                "invalid",
                [1.0, 0.0],
                "invalid",
                [[1.0, 0.0], [1.0]],
                ["first", "second"]
            )
        } catch (_: Exception) {
            rejected = true
        }
        @Assert(rejected, true)
    }
}

private func testGroup(name: String, x: Float64, y: Float64): VectorGroup {
    return VectorGroup(
        name,
        [x, y],
        name,
        [[x, y], [x + 0.01, y + 0.01]],
        ["${name}-0", "${name}-1"]
    )
}

private func testKeyVectors(): HashMap<String, Array<Float64>> {
    let values = HashMap<String, Array<Float64>>()
    values["left"] = [-1.0, 0.0]
    values["right"] = [1.0, 0.0]
    return values
}

private func testDescription(): HashMap<String, Any> {
    let description = HashMap<String, Any>()
    description["key"] = "left"
    description["group_name"] = "left-group"
    description["text"] = "inserted"
    return description
}

@Test
public class VectorMapIndexLifecycleTest {
    @TestCase
    public func invalidatesAndRebuildsHdmgAfterMutation(): Unit {
        let map = VectorMap()
        map.insert("left", testGroup("left-group", -1.0, 0.0))
        map.insert("right", testGroup("right-group", 1.0, 0.0))
        map.setKeyVectors(testKeyVectors())
        map.buildIndex()
        map.buildHdmgWithConfig(HdmgBuildConfig(
            embeddingK: 1,
            semanticIntraK: 1,
            semanticBridgeKeys: 1,
            semanticBridgePerKey: 1,
            useMutualEmbedding: false
        ))

        let built = map.getIndexState()
        @Assert((built["rep_index_current"] as Bool).getOrThrow(), true)
        @Assert((built["single_index_current"] as Bool).getOrThrow(), true)
        @Assert((built["hdmg_current"] as Bool).getOrThrow(), true)

        @Assert(map.addVector([-0.9, 0.0], testDescription()) >= 0, true)

        let stale = map.getIndexState()
        @Assert((stale["rep_index_current"] as Bool).getOrThrow(), false)
        @Assert((stale["single_index_current"] as Bool).getOrThrow(), false)
        @Assert((stale["hdmg_current"] as Bool).getOrThrow(), false)

        let fallback = map.searchHdmg(
            [-1.0, 0.0],
            queryKeyVector: Some([-1.0, 0.0]),
            alpha: 0.5,
            topK: 2
        )
        @Assert(fallback.size, 2)
        let stats = map.getLastHdmgSearchStats()
        @Assert((stats["mode"] as String).getOrThrow(), "fallback")
    }

    @TestCase
    public func returnsStableTopKForEqualDistances(): Unit {
        let map = VectorMap()
        map.insert("b", testGroup("b-group", 1.0, 0.0))
        map.insert("a", testGroup("a-group", 1.0, 0.0))
        let first = map.search([0.0, 1.0], topK: 4, mode: "single")
        let second = map.search([0.0, 1.0], topK: 4, mode: "single")
        @Assert(first.size, second.size)
        for (i in 0..first.size) {
            @Assert(first[i].key, second[i].key)
            @Assert(first[i].vectorIdx, second[i].vectorIdx)
        }
    }
}

@Test
public class HdmgConfigurationTest {
    @TestCase
    public func rejectsInvalidBuildAndSearchParameters(): Unit {
        var buildRejected = false
        try {
            let _ = HdmgBuildConfig(embeddingK: -1)
        } catch (_: IllegalArgumentException) {
            buildRejected = true
        }
        @Assert(buildRejected, true)

        var searchRejected = false
        try {
            let _ = HdmgSearchConfig(clusterPoolMultiplier: 0)
        } catch (_: IllegalArgumentException) {
            searchRejected = true
        }
        @Assert(searchRejected, true)
    }
}
