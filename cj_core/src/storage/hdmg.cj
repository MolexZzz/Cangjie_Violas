package semanticvg_cj.storage

public class HdmgScore <: Comparable<HdmgScore> {
    public let score: Float64
    public let nodeId: Int64

    public init(score: Float64, nodeId: Int64) { this.score = score; this.nodeId = nodeId }

    public func compare(other: HdmgScore): Ordering {
        if (this.score < other.score) { return Ordering.LT }
        if (this.score > other.score) { return Ordering.GT }
        if (this.nodeId < other.nodeId) { return Ordering.LT }
        if (this.nodeId > other.nodeId) { return Ordering.GT }
        return Ordering.EQ
    }
}

public class HdmgBuildConfig {
    public let embeddingK: Int64
    public let semanticIntraK: Int64
    public let semanticBridgeKeys: Int64
    public let semanticBridgePerKey: Int64
    public let useMutualEmbedding: Bool
    public let assumeNormalized: Bool

    public init(embeddingK!: Int64 = 16, semanticIntraK!: Int64 = 4,
        semanticBridgeKeys!: Int64 = 2, semanticBridgePerKey!: Int64 = 1,
        useMutualEmbedding!: Bool = true, assumeNormalized!: Bool = false) {
        if (embeddingK < 0 || semanticIntraK < 0 || semanticBridgeKeys < 0 || semanticBridgePerKey < 0) {
            throw IllegalArgumentException("HDMG 邻居参数不能为负数")
        }
        this.embeddingK = embeddingK
        this.semanticIntraK = semanticIntraK
        this.semanticBridgeKeys = semanticBridgeKeys
        this.semanticBridgePerKey = semanticBridgePerKey
        this.useMutualEmbedding = useMutualEmbedding
        this.assumeNormalized = assumeNormalized
    }
}

public class HdmgSearchConfig {
    public let maxSteps: Int64
    public let entryAlphaThreshold: Float64
    public let clusterPoolMultiplier: Int64
    public let topKeyCandidates: Int64
    public let extraHops: Int64

    public init(maxSteps!: Int64 = 100, entryAlphaThreshold!: Float64 = 0.3,
        clusterPoolMultiplier!: Int64 = 3, topKeyCandidates!: Int64 = 5, extraHops!: Int64 = 0) {
        if (maxSteps < 0 || clusterPoolMultiplier < 1 || topKeyCandidates < 1 || extraHops < 0) {
            throw IllegalArgumentException("HDMG 搜索配置非法")
        }
        this.maxSteps = maxSteps
        this.entryAlphaThreshold = clampUnit(entryAlphaThreshold)
        this.clusterPoolMultiplier = clusterPoolMultiplier
        this.topKeyCandidates = topKeyCandidates
        this.extraHops = extraHops
    }
}
