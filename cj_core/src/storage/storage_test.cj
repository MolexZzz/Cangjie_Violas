package semanticvg_cj.storage

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
}
