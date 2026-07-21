package semanticvg_cj.storage

import std.collection.*
import std.math.*

public func clusterCount(n: Int64, alpha: Float64): Int64 {
    if (n <= 0) { return 0 }
    let a = clampUnit(alpha)
    var count = Int64(pow(Float64(n), 1.0 - a) + 0.5)
    if (count < 1) { count = 1 }
    if (count > n) { count = n }
    return count
}

public func clusterSuffix(n: Int64): String {
    let s = n.toString()
    if (s.size >= 4) { return s }
    var out = s
    for (_ in 0..(4 - s.size)) { out = "0" + out }
    return out
}

func squaredEuclidean(a: Array<Float64>, b: Array<Float64>): Float64 {
    if (a.size != b.size) {
        throw IllegalArgumentException("向量维度不一致: ${a.size} != ${b.size}")
    }
    var sum = 0.0
    for (i in 0..a.size) {
        let d = a[i] - b[i]
        sum += d * d
    }
    return sum
}

// 确定性 Lloyd KMeans：核心算法独立于 VectorMap，便于替换为其他聚类 backend。
public func deterministicKMeans(
    vectors: ArrayList<Array<Float64>>,
    nClusters: Int64,
    maxIterations!: Int64 = 100
): Array<Int64> {
    let n = vectors.size
    if (nClusters <= 0 || nClusters > n) {
        throw IllegalArgumentException("非法聚类数量: ${nClusters}, vector_count=${n}")
    }
    let dim = vectors[0].size
    for (vec in vectors) {
        if (vec.size != dim) { throw IllegalArgumentException("组内向量维度不一致: ${vec.size} != ${dim}") }
    }

    let centroids = ArrayList<Array<Float64>>()
    centroids.add(Array<Float64>(dim, { d => vectors[0][d] }))
    while (centroids.size < nClusters) {
        var bestIdx: Int64 = 0
        var bestDistance = -1.0
        for (i in 0..n) {
            var nearest = Float64.Max
            for (centroid in centroids) {
                let d = squaredEuclidean(vectors[i], centroid)
                if (d < nearest) { nearest = d }
            }
            if (nearest > bestDistance) { bestDistance = nearest; bestIdx = i }
        }
        let selectedIdx = bestIdx
        centroids.add(Array<Float64>(dim, { d => vectors[selectedIdx][d] }))
    }

    let labels = Array<Int64>(n, repeat: -1)
    var iteration: Int64 = 0
    while (iteration < maxIterations) {
        var changed = false
        for (i in 0..n) {
            var bestCluster: Int64 = 0
            var bestDistance = squaredEuclidean(vectors[i], centroids[0])
            for (c in 1..nClusters) {
                let d = squaredEuclidean(vectors[i], centroids[c])
                if (d < bestDistance) { bestDistance = d; bestCluster = c }
            }
            if (labels[i] != bestCluster) { labels[i] = bestCluster; changed = true }
        }
        let sums = Array<Array<Float64>>(nClusters, { _ => Array<Float64>(dim, repeat: 0.0) })
        let counts = Array<Int64>(nClusters, repeat: 0)
        for (i in 0..n) {
            let c = labels[i]
            counts[c]++
            for (d in 0..dim) { sums[c][d] += vectors[i][d] }
        }
        for (c in 0..nClusters) {
            if (counts[c] > 0) {
                let denom = Float64(counts[c])
                centroids[c] = Array<Float64>(dim, { d => sums[c][d] / denom })
            }
        }
        iteration++
        if (!changed) { break }
    }
    return labels
}

// NumPy RandomState 使用的 MT19937 与 53-bit random_sample。
class Mt19937 {
    private let state: Array<UInt32>
    private var index: Int64

    init(seed: UInt32) {
        this.state = Array<UInt32>(624, repeat: UInt32(0))
        this.state[0] = seed
        for (i in 1..624) {
            let previous = this.state[i - 1]
            let mixed = UInt64(previous ^ (previous >> 30))
            this.state[i] = UInt32((UInt64(1812433253) * mixed + UInt64(i)) & UInt64(0xffffffff))
        }
        this.index = 624
    }

    private func twist(): Unit {
        let upper = UInt32(0x80000000)
        let lower = UInt32(0x7fffffff)
        for (i in 0..624) {
            let y = (this.state[i] & upper) | (this.state[(i + 1) % 624] & lower)
            var next = this.state[(i + 397) % 624] ^ (y >> 1)
            if ((y & UInt32(1)) != UInt32(0)) { next = next ^ UInt32(0x9908b0df) }
            this.state[i] = next
        }
        this.index = 0
    }

    private func nextUInt32(): UInt32 {
        if (this.index >= 624) { twist() }
        var y = this.state[this.index]
        this.index++
        y = y ^ (y >> 11)
        y = y ^ ((y << 7) & UInt32(0x9d2c5680))
        y = y ^ ((y << 15) & UInt32(0xefc60000))
        y = y ^ (y >> 18)
        return y
    }

    func randomDouble(): Float64 {
        let a = UInt64(nextUInt32() >> 5)
        let b = UInt64(nextUInt32() >> 6)
        return (Float64(a) * 67108864.0 + Float64(b)) / 9007199254740992.0
    }
}

func meanOfRows(vectors: ArrayList<Array<Float64>>): Array<Float64> {
    let dim = vectors[0].size
    let mean = Array<Float64>(dim, repeat: 0.0)
    for (vector in vectors) {
        for (d in 0..dim) { mean[d] += vector[d] }
    }
    for (d in 0..dim) { mean[d] /= Float64(vectors.size) }
    return mean
}

func kmeansTolerance(vectors: ArrayList<Array<Float64>>): Float64 {
    let mean = meanOfRows(vectors)
    var totalVariance = 0.0
    for (d in 0..mean.size) {
        var variance = 0.0
        for (vector in vectors) {
            let delta = vector[d] - mean[d]
            variance += delta * delta
        }
        totalVariance += variance / Float64(vectors.size)
    }
    return (totalVariance / Float64(mean.size)) * 0.0001
}

func kmeansPlusPlus(
    vectors: ArrayList<Array<Float64>>,
    nClusters: Int64,
    random: Mt19937
): ArrayList<Array<Float64>> {
    let n = vectors.size
    let dim = vectors[0].size
    let centers = ArrayList<Array<Float64>>()
    var first = Int64(random.randomDouble() * Float64(n))
    if (first >= n) { first = n - 1 }
    let firstIndex = first
    centers.add(Array<Float64>(dim, { d => vectors[firstIndex][d] }))
    let closest = Array<Float64>(n, { i => squaredEuclidean(vectors[i], centers[0]) })
    var potential = 0.0
    for (distance in closest) { potential += distance }
    let localTrials = 2 + Int64(log(Float64(nClusters)))

    while (centers.size < nClusters) {
        var bestCandidate: Int64 = 0
        var bestPotential = Float64.Max
        var bestDistances = Array<Float64>()
        for (_ in 0..localTrials) {
            let target = random.randomDouble() * potential
            var cumulative = 0.0
            var candidate: Int64 = n - 1
            for (i in 0..n) {
                cumulative += closest[i]
                if (target <= cumulative) { candidate = i; break }
            }
            let distances = Array<Float64>(n, repeat: 0.0)
            var candidatePotential = 0.0
            for (i in 0..n) {
                let distance = squaredEuclidean(vectors[i], vectors[candidate])
                distances[i] = if (distance < closest[i]) { distance } else { closest[i] }
                candidatePotential += distances[i]
            }
            if (candidatePotential < bestPotential) {
                bestCandidate = candidate
                bestPotential = candidatePotential
                bestDistances = distances
            }
        }
        let selectedCandidate = bestCandidate
        centers.add(Array<Float64>(dim, { d => vectors[selectedCandidate][d] }))
        for (i in 0..n) { closest[i] = bestDistances[i] }
        potential = bestPotential
    }
    return centers
}

func singleLloydRun(
    vectors: ArrayList<Array<Float64>>,
    initialCenters: ArrayList<Array<Float64>>,
    tolerance: Float64,
    maxIterations: Int64
): (Array<Int64>, Float64) {
    let n = vectors.size
    let k = initialCenters.size
    let dim = vectors[0].size
    var centers = ArrayList<Array<Float64>>()
    for (center in initialCenters) { centers.add(Array<Float64>(dim, { d => center[d] })) }
    let labels = Array<Int64>(n, repeat: -1)
    var iteration: Int64 = 0
    while (iteration < maxIterations) {
        let oldLabels = Array<Int64>(n, { i => labels[i] })
        for (i in 0..n) {
            var best: Int64 = 0
            var bestDistance = squaredEuclidean(vectors[i], centers[0])
            for (c in 1..k) {
                let distance = squaredEuclidean(vectors[i], centers[c])
                if (distance < bestDistance) { bestDistance = distance; best = c }
            }
            labels[i] = best
        }
        var strictConvergence = true
        for (i in 0..n) { if (labels[i] != oldLabels[i]) { strictConvergence = false; break } }

        let sums = Array<Array<Float64>>(k, { _ => Array<Float64>(dim, repeat: 0.0) })
        let counts = Array<Int64>(k, repeat: 0)
        for (i in 0..n) {
            counts[labels[i]]++
            for (d in 0..dim) { sums[labels[i]][d] += vectors[i][d] }
        }
        // 对齐 sklearn Lloyd 的空簇重定位：把最远样本从原簇移动到空簇，并同步修正原簇计数/向量和。
        let relocated = ArrayList<Int64>()
        for (c in 0..k) {
            if (counts[c] == 0) {
                var farthest: Int64 = 0
                var farthestDistance = -1.0
                for (i in 0..n) {
                    if (!relocated.contains(i)) {
                        let distance = squaredEuclidean(vectors[i], centers[labels[i]])
                        if (distance > farthestDistance) { farthestDistance = distance; farthest = i }
                    }
                }
                relocated.add(farthest)
                let sourceCluster = labels[farthest]
                counts[sourceCluster]--
                for (d in 0..dim) { sums[sourceCluster][d] -= vectors[farthest][d] }
                let farthestIndex = farthest
                sums[c] = Array<Float64>(dim, { d => vectors[farthestIndex][d] })
                counts[c] = 1
            }
        }
        let nextCenters = ArrayList<Array<Float64>>()
        var centerShift = 0.0
        for (c in 0..k) {
            let next = Array<Float64>(dim, { d => sums[c][d] / Float64(counts[c]) })
            centerShift += squaredEuclidean(centers[c], next)
            nextCenters.add(next)
        }
        centers = nextCenters
        iteration++
        if (strictConvergence || centerShift <= tolerance) { break }
    }

    var inertia = 0.0
    for (i in 0..n) {
        var best: Int64 = 0
        var bestDistance = squaredEuclidean(vectors[i], centers[0])
        for (c in 1..k) {
            let distance = squaredEuclidean(vectors[i], centers[c])
            if (distance < bestDistance) { bestDistance = distance; best = c }
        }
        labels[i] = best
        inertia += bestDistance
    }
    return (labels, inertia)
}

// 对齐 sklearn KMeans(random_state=42, n_init=10) 的关键语义：MT19937、KMeans++、10次取最优。
public func sklearnCompatibleKMeans(
    vectors: ArrayList<Array<Float64>>,
    nClusters: Int64,
    nInit!: Int64 = 10,
    maxIterations!: Int64 = 300
): Array<Int64> {
    if (nClusters <= 0 || nClusters > vectors.size || nInit <= 0) {
        throw IllegalArgumentException("非法 sklearn-compatible KMeans 参数")
    }
    let dim = vectors[0].size
    for (vector in vectors) {
        if (vector.size != dim) { throw IllegalArgumentException("组内向量维度不一致") }
    }
    let mean = meanOfRows(vectors)
    let centered = ArrayList<Array<Float64>>()
    for (vector in vectors) { centered.add(Array<Float64>(dim, { d => vector[d] - mean[d] })) }
    let tolerance = kmeansTolerance(centered)
    let random = Mt19937(UInt32(42))
    var bestLabels = Array<Int64>()
    var bestInertia = Float64.Max
    for (_ in 0..nInit) {
        let initial = kmeansPlusPlus(centered, nClusters, random)
        let (labels, inertia) = singleLloydRun(centered, initial, tolerance, maxIterations)
        if (inertia < bestInertia) { bestInertia = inertia; bestLabels = labels }
    }
    return bestLabels
}
