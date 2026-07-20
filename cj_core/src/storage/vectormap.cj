package semanticvg_cj.storage

import std.collection.*
import std.time.*
import std.random.*
import std.convert.* 
import std.sort.* 
import std.math.*

// ============= 检索结果实体 =============

public class SearchResult <: Comparable<SearchResult> & ToString {
    public let distance: Float64
    public let key: String
    public let group: VectorGroup
    public let vectorIdx: Option<Int64>

    public init(distance: Float64, key: String, group: VectorGroup, vectorIdx: Option<Int64>) {
        this.distance = distance
        this.key = key
        this.group = group
        this.vectorIdx = vectorIdx
    }

    public func compare(other: SearchResult): Ordering {
        if (this.distance < other.distance) { return Ordering.LT }
        if (this.distance > other.distance) { return Ordering.GT }
        return Ordering.EQ
    }

    public func toString(): String {
        let idxStr = match(this.vectorIdx) {
            case Some(idx) => idx.toString()
            case None => "None"
        }
        return "SearchResult(distance=${this.distance}, key=${this.key}, group=${this.group.groupName}, vector_idx=${idxStr})"
    }
}

public class VectorMapEntry {
    public var metadata: HashMap<String, Any>
    public var groups: ArrayList<VectorGroup>
    
    public init(metadata: HashMap<String, Any>) {
        this.metadata = metadata
        this.groups = ArrayList<VectorGroup>()
    }
}

private class HdmgScore <: Comparable<HdmgScore> {
    public let score: Float64
    public let nodeId: Int64

    public init(score: Float64, nodeId: Int64) {
        this.score = score
        this.nodeId = nodeId
    }

    public func compare(other: HdmgScore): Ordering {
        if (this.score < other.score) { return Ordering.LT }
        if (this.score > other.score) { return Ordering.GT }
        return Ordering.EQ
    }
}

private func _nowNs(): Int64 {
    return DateTime.now().toUnixTimeStamp().toNanoseconds()
}

// ============= 向量映射表 =============

public class VectorMap <: ToString {
    public var data: HashMap<String, VectorMapEntry>
    public var contextMap: HashMap<String, String>
    public var keyVectors: HashMap<String, Array<Float64>>
    private var hdmgBuilt: Bool
    private var hdmgLastSearchStats: HashMap<String, Any>
    private var hdmgNodeKeys: ArrayList<String>
    private var hdmgNodeGroups: ArrayList<VectorGroup>
    private var hdmgBaseKeys: ArrayList<String>
    private var hdmgEmbeddingEdges: ArrayList<ArrayList<Int64>>
    private var hdmgSemanticEdges: ArrayList<ArrayList<Int64>>
    private var hdmgKeyEntries: HashMap<String, Int64>
    private var hdmgGlobalEntry: Option<Int64>

    public init() {
        this.data = HashMap<String, VectorMapEntry>()
        this.contextMap = HashMap<String, String>()
        this.keyVectors = HashMap<String, Array<Float64>>()
        this.hdmgBuilt = false
        this.hdmgLastSearchStats = HashMap<String, Any>()
        this.hdmgNodeKeys = ArrayList<String>()
        this.hdmgNodeGroups = ArrayList<VectorGroup>()
        this.hdmgBaseKeys = ArrayList<String>()
        this.hdmgEmbeddingEdges = ArrayList<ArrayList<Int64>>()
        this.hdmgSemanticEdges = ArrayList<ArrayList<Int64>>()
        this.hdmgKeyEntries = HashMap<String, Int64>()
        this.hdmgGlobalEntry = None
    }

    private func _getKeysAsArray(): Array<String> {
        let keysList = ArrayList<String>() // 使用 let 保证闭包捕获安全
        for ((k, _) in this.data) { keysList.add(k) }
        return Array<String>(keysList.size, { i => keysList[i] })
    }

    // ============= 插入向量组、添加向量 ============= 

    // 向指定的Key中插入一个完整的向量组
    public func insert(key: String, vectorGroup: VectorGroup, metadataOpt!: Option<HashMap<String, Any>> = None): Unit {
        if (!this.data.contains(key)) {
            let meta = match(metadataOpt) {
                case Some(m) => m
                case None => HashMap<String, Any>()
            }
            this.data[key] = VectorMapEntry(meta)
        }
        let entry = this.data[key]

        if (!entry.metadata.contains("vector_types")) {
            entry.metadata["vector_types"] = ArrayList<String>()
        }
        let vTypesOpt = entry.metadata["vector_types"] as ArrayList<String>
        let vTypes = vTypesOpt.getOrThrow()
        if (!vTypes.contains(vectorGroup.vectorType)) { vTypes.add(vectorGroup.vectorType) }

        if (!entry.metadata.contains("group_types")) {
            entry.metadata["group_types"] = ArrayList<String>()
        }
        let gTypesOpt = entry.metadata["group_types"] as ArrayList<String>
        let gTypes = gTypesOpt.getOrThrow()
        if (!gTypes.contains(vectorGroup.groupType)) { gTypes.add(vectorGroup.groupType) }

        if (!entry.metadata.contains("relations")) {
            let rels = HashMap<String, Any>()
            rels["pair"] = ArrayList<HashMap<String, Any>>()
            rels["tree"] = ArrayList<HashMap<String, Any>>()
            entry.metadata["relations"] = rels
        }

        var replaceIdx = -1
        for (i in 0..entry.groups.size) {
            if (entry.groups[i].groupName == vectorGroup.groupName) {
                println("警告: 组名 '${vectorGroup.groupName}' 已存在，将覆盖")
                replaceIdx = i
                break
            }
        }
        
        if (replaceIdx >= 0) { 
            let newGroups = ArrayList<VectorGroup>()
            for (i in 0..entry.groups.size) {
                if (i != replaceIdx) { newGroups.add(entry.groups[i]) }
            }
            entry.groups = newGroups
        }
        entry.groups.add(vectorGroup)
        this.hdmgBuilt = false
    }

    private func _baseKey(key: String): String {
        let idxOpt = key.lastIndexOf("-")
        if (idxOpt.isNone()) { return key }
        let idx = idxOpt.getOrThrow()
        if (idx <= 0 || idx >= key.size - 1) { return key }
        let suffix = key[(idx + 1)..key.size]
        try {
            let _ = Int64.parse(suffix)
            return key[0..idx]
        } catch (e: Exception) {
            return key
        }
    }

    private func _meanVector(vectors: Array<Array<Float64>>): Array<Float64> {
        if (vectors.isEmpty()) { return Array<Float64>() }
        let dim = vectors[0].size
        let out = Array<Float64>(dim, repeat: 0.0)
        for (vec in vectors) {
            for (i in 0..dim) { out[i] += vec[i] }
        }
        let denom = Float64(vectors.size)
        for (i in 0..dim) { out[i] = out[i] / denom }
        return out
    }

    private func _distanceFunc(distanceMethod: String): (Array<Float64>, Array<Float64>) -> Float64 {
        if (distanceMethod == "euclidean") { return euclideanDistance }
        if (distanceMethod == "manhattan") { return manhattanDistance }
        return cosineDistance
    }

    private func _clusterSuffix(n: Int64): String {
        let s = n.toString()
        if (s.size >= 4) { return s }
        var out = s
        for (_ in 0..(4 - s.size)) { out = "0" + out }
        return out
    }

    private func _expandedSearchKeys(keyOpt: Option<String>): Array<String> {
        match (keyOpt) {
            case None => return _getKeysAsArray()
            case Some(k) =>
                if (this.data.contains(k)) { return [k] }
                let keys = ArrayList<String>()
                let prefix = k + "-"
                for ((dk, _) in this.data) {
                    if (dk.startsWith(prefix)) { keys.add(dk) }
                }
                return keys.toArray()
        }
    }

    public func setKeyVectors(keyVectors: HashMap<String, Array<Float64>>): Unit {
        this.keyVectors = keyVectors
        this.hdmgBuilt = false
    }

    public func getKeyVector(key: String): Option<Array<Float64>> {
        let base = _baseKey(key)
        if (this.keyVectors.contains(base)) { return Some(this.keyVectors[base]) }
        if (this.keyVectors.contains(key)) { return Some(this.keyVectors[key]) }
        return None
    }

    public func insertWithAutoCluster(
        key: String,
        vectorGroup: VectorGroup,
        metadataOpt!: Option<HashMap<String, Any>> = None,
        alpha!: Float64 = 0.5
    ): Int64 {
        let n = vectorGroup.vectors.size
        if (n == 0) { return 0 }

        let a = if (alpha < 0.0) { 0.0 } else if (alpha > 1.0) { 1.0 } else { alpha }
        var nClusters: Int64 = 1
        if (a <= 0.05) {
            nClusters = n
        } else if (a >= 0.95) {
            nClusters = 1
        } else {
            nClusters = Int64(sqrt(Float64(n)) + 0.5)
            if (nClusters < 1) { nClusters = 1 }
            if (nClusters > n) { nClusters = n }
        }

        if (nClusters == 1 || n < 5) {
            insert("${key}-0001", vectorGroup, metadataOpt: metadataOpt)
            return 1
        }

        var created: Int64 = 0
        var start: Int64 = 0
        while (start < n) {
            let remaining = n - start
            let remainingClusters = nClusters - created
            var chunk = remaining / remainingClusters
            if (remaining % remainingClusters != 0) { chunk++ }
            let end = if (start + chunk > n) { n } else { start + chunk }
            let size = end - start
            let startIdx = start
            let vecArr = Array<Array<Float64>>(size, { i => vectorGroup.vectors[startIdx + i] })
            let descArr = Array<Any>(size, { i => vectorGroup.descriptions[startIdx + i] })
            let rep = _meanVector(vecArr)
            let subGroup = VectorGroup(
                "${vectorGroup.groupName}_cluster_${created + 1}",
                rep,
                "cluster_mean_${created + 1}",
                vecArr,
                descArr,
                vectorType: vectorGroup.vectorType,
                groupType: vectorGroup.groupType
            )
            let subKey = "${key}-${_clusterSuffix(created + 1)}"
            insert(subKey, subGroup, metadataOpt: metadataOpt)
            created++
            start = end
        }
        return created
    }

    public func buildIndex(): Unit { buildRepIndex(); buildSingleIndex() }
    public func buildRepIndex(): Unit { /* 仓颉版当前使用精确遍历；保留 API 对齐 Python。 */ }
    public func buildSingleIndex(): Unit { /* 仓颉版当前使用精确遍历；保留 API 对齐 Python。 */ }

    // 向已有的特定组中追加一个向量
    public func addVector(vector: Array<Float64>, description: HashMap<String, Any>): Int64 {
        let keyOpt = description["key"] as String
        let groupNameOpt = description["group_name"] as String
        if (keyOpt.isNone() || groupNameOpt.isNone()) { return -1 }

        let groupOpt = getGroupByName(keyOpt.getOrThrow(), groupNameOpt.getOrThrow())
        match (groupOpt) {
            case None => 
                println("warning: vector_map 没有找到 key '${keyOpt.getOrThrow()}' 和 group_name '${groupNameOpt.getOrThrow()}'")
                return -1
            case Some(group) => 
                return group.append(vector, description)
        }
    }

    // 批量插入向量
    public func addVectorList(vectorList: Array<Array<Float64>>, descriptionList: Array<HashMap<String, Any>>): Int64 {
        let globalId = "uuid_${DateTime.now().toUnixTimeStamp()}_${Random().nextInt64()}"
        var idx = 0

        for (i in 0..vectorList.size) {
            let vec = vectorList[i]
            let desc = descriptionList[i]
            let normalizedDesc = normalizeDescription(desc)
            
            let curId = "${globalId}_${idx}"
            normalizedDesc["id"] = curId
            
            let insertIdx = addVector(vec, normalizedDesc)
            if (insertIdx >= 0) {
                let keyStr = normalizedDesc["key"] as String
                let groupNameStr = normalizedDesc["group_name"] as String
                this.contextMap[curId] = "${keyStr.getOrThrow()}%${groupNameStr.getOrThrow()}%${insertIdx}"
                idx++
            }
        }
        return Int64(idx)
    }

    // ============= 关系建模方法 =============

    // 建立两个向量组之间1 ：1 的配对关系（跨模态）
    public func addPairRelation(key: String, group1Id: String, group2Id: String, relationType!: String = "paired"): Bool {
        if (!this.data.contains(key)) { return false }
        
        var group1Opt: Option<VectorGroup> = None
        var group2Opt: Option<VectorGroup> = None
        
        for (g in this.data[key].groups) {
            if (g.groupId == group1Id) { group1Opt = Some(g) }
            if (g.groupId == group2Id) { group2Opt = Some(g) }
        }
        
        if (group1Opt.isNone() || group2Opt.isNone()) {
            println("warning: 找不到指定的向量组 ${group1Id} 或 ${group2Id}")
            return false
        }
        
        let g1 = group1Opt.getOrThrow()
        let g2 = group2Opt.getOrThrow()
        
        if (g1.size() != g2.size()) {
            println("警告: 配对关系要求向量组大小相等")
            return false
        }
        
        let relation = HashMap<String, Any>()
        relation["group1_id"] = group1Id
        relation["group2_id"] = group2Id
        relation["group1_name"] = g1.groupName
        relation["group2_name"] = g2.groupName
        relation["relation_type"] = relationType
        relation["vector_count"] = g1.size()
        
        let rels = this.data[key].metadata["relations"] as HashMap<String, Any>
        let pairList = rels.getOrThrow()["pair"] as ArrayList<HashMap<String, Any>>
        pairList.getOrThrow().add(relation)
        return true
    }

    // 建立父子结点关系
    public func addTreeRelation(
        key: String, parentName: String, childName: String, 
        parentId!: Option<String> = None, childId!: Option<String> = None,
        relationData!: Option<HashMap<String, Any>> = None
    ): Bool {
        if (!this.data.contains(key)) { return false }
        
        let relation = HashMap<String, Any>()
        relation["parent_name"] = parentName
        relation["child_name"] = childName
        match(parentId) { case Some(id) => relation["parent_id"] = id; case None => () }
        match(childId) { case Some(id) => relation["child_id"] = id; case None => () }
        
        match(relationData) {
            case Some(data) => 
                for ((k, v) in data) { relation[k] = v }
            case None => ()
        }
        
        let rels = this.data[key].metadata["relations"] as HashMap<String, Any>
        let treeList = rels.getOrThrow()["tree"] as ArrayList<HashMap<String, Any>>
        treeList.getOrThrow().add(relation)
        return true
    }

    // ============= 查询方法 =============
    public func get(key: String, returnType!: String = "all"): Any {
        if (!this.data.contains(key)) { return Option<Any>.None }
        if (returnType == "metadata") { return this.data[key].metadata }
        if (returnType == "groups") { return this.data[key].groups }
        return this.data[key]
    }

    public func getAllKeys(): Array<String> {
        return _getKeysAsArray()
    }

    public func getGroupByName(key: String, groupName: String): Option<VectorGroup> {
        if (this.data.contains(key)) {
            for (g in this.data[key].groups) {
                if (g.groupName == groupName) { return Some(g) }
            }
        }
        return None
    }

    public func getGroupById(key: String, groupId: String): Option<VectorGroup> {
        if (this.data.contains(key)) {
            for (g in this.data[key].groups) {
                if (g.groupId == groupId) { return Some(g) }
            }
        }
        return None
    }

    public func getRelations(key: String): HashMap<String, Any> {
        if (this.data.contains(key)) {
            let relsOpt = this.data[key].metadata["relations"] as HashMap<String, Any>
            if (relsOpt.isSome()) { return relsOpt.getOrThrow() }
        }
        return HashMap<String, Any>()
    }

    public func getHierarchyTree(key: String): HashMap<String, Any> {
        let hierarchy = HashMap<String, Any>()
        let rootNodes = ArrayList<String>()
        let parentChildren = HashMap<String, ArrayList<String>>()
        let nodeGroups = HashMap<String, ArrayList<String>>()
        
        hierarchy["root_nodes"] = rootNodes
        hierarchy["parent_children"] = parentChildren
        hierarchy["node_groups"] = nodeGroups

        if (!this.data.contains(key)) { return hierarchy }
        let relations = getRelations(key)
        
        let treeListOpt = relations["tree"] as ArrayList<HashMap<String, Any>>
        if (treeListOpt.isNone()) { return hierarchy }
        let treeList = treeListOpt.getOrThrow()

        for (treeRel in treeList) {
            let parentName = (treeRel["parent_name"] as String).getOrThrow()
            let childName = (treeRel["child_name"] as String).getOrThrow()
            
            let parentIdOpt = if (treeRel.contains("parent_id")) { treeRel["parent_id"] as String } else { Option<String>.None }
            let childIdOpt = if (treeRel.contains("child_id")) { treeRel["child_id"] as String } else { Option<String>.None }
            
            if (!parentChildren.contains(parentName)) { parentChildren[parentName] = ArrayList<String>() }
            parentChildren[parentName].add(childName)

            if (parentIdOpt.isSome()) {
                let pId = parentIdOpt.getOrThrow()
                if (!nodeGroups.contains(parentName)) { nodeGroups[parentName] = ArrayList<String>() }
                if (!nodeGroups[parentName].contains(pId)) { nodeGroups[parentName].add(pId) }
            }

            if (childIdOpt.isSome()) {
                let cId = childIdOpt.getOrThrow()
                if (!nodeGroups.contains(childName)) { nodeGroups[childName] = ArrayList<String>() }
                if (!nodeGroups[childName].contains(cId)) { nodeGroups[childName].add(cId) }
            }
        }

        let allChildren = ArrayList<String>()
        for ((_, children) in parentChildren) {
            for (c in children) { 
                if (!allChildren.contains(c)) { allChildren.add(c) } 
            }
        }

        for (node in parentChildren.keys()) {
            if (!allChildren.contains(node)) { rootNodes.add(node) }
        }

        return hierarchy
    }

    // ============= 检索与召回 =============
    
    // 核心单向量检索：遍历所有匹配条件的组进行距离计算，最后全局排序截断 topK
    // mode：检索模式（逐个对比 single、只看中心点 representative、还是算平均值 group1）
    // distanceMethod：余弦 cosine，欧氏 euclidean，曼哈顿 manhattan
    public func search(
        queryVector: Array<Float64>, groupName!: Option<String> = None, topK!: Int64 = 5, 
        mode!: String = "single", distanceMethod!: String = "cosine",
        keyOpt!: Option<String> = None, vectorType!: Option<String> = None, groupType!: Option<String> = None
    ): Array<SearchResult> {
        let results = ArrayList<SearchResult>() 
        let searchKeys = _expandedSearchKeys(keyOpt)

        let distFunc = if (distanceMethod == "euclidean") { euclideanDistance } 
                       else if (distanceMethod == "manhattan") { manhattanDistance } 
                       else { cosineDistance }

        for (k in searchKeys) {
            if (!this.data.contains(k)) { continue }
            for (group in this.data[k].groups) {
                if (groupName.isSome() && group.groupName != groupName.getOrThrow()) { continue }
                if (vectorType.isSome() && group.vectorType != vectorType.getOrThrow()) { continue }
                if (groupType.isSome() && group.groupType != groupType.getOrThrow()) { continue }

                if (mode == "single") {
                    for (i in 0..group.vectors.size) {
                        let dist = distFunc(queryVector, group.vectors[i])
                        results.add(SearchResult(dist, k, group, Some(i)))
                    }
                } else if (mode == "representative") {
                    let dist = distFunc(queryVector, group.representative)
                    results.add(SearchResult(dist, k, group, None))
                } else if (mode == "group1") {
                    var totalDist = 0.0
                    for (vec in group.vectors) { totalDist += distFunc(queryVector, vec) }
                    let avgDist = if (group.vectors.size > 0) { totalDist / Float64(group.vectors.size) } else { 0.0 }
                    results.add(SearchResult(avgDist, k, group, None))
                }
            }
        }
        
         sort(results, by: {a, b => a.compare(b)})
        
        // 使用原生的比较运算代替 std.math.min 以消除 warning
        let limit = if (topK < results.size) { topK } else { results.size }
        return Array<SearchResult>(limit, {i => results[i]})
    }

    // 多向量查询（使用一个查询向量矩阵）
    // 检索模式：groupN
    // groupNDistanceMethod：hausdorff、maxSim

    public func searchMulti(
        queryVectors: Array<Array<Float64>>, groupName!: Option<String> = None, topK!: Int64 = 5, 
        groupNDistanceMethod!: String = "hausdorff", distanceMethod!: String = "cosine",
        keyOpt!: Option<String> = None, vectorType!: Option<String> = None, groupType!: Option<String> = None
    ): Array<SearchResult> {
        let results = ArrayList<SearchResult>() 
        let searchKeys = _expandedSearchKeys(keyOpt)
        
        let distFunc = if (distanceMethod == "euclidean") { euclideanDistance } 
                       else if (distanceMethod == "manhattan") { manhattanDistance } 
                       else { cosineDistance }

        for (k in searchKeys) {
            if (!this.data.contains(k)) { continue }
            for (group in this.data[k].groups) {
                if (groupName.isSome() && group.groupName != groupName.getOrThrow()) { continue }
                if (vectorType.isSome() && group.vectorType != vectorType.getOrThrow()) { continue }
                if (groupType.isSome() && group.groupType != groupType.getOrThrow()) { continue }

                if (groupNDistanceMethod == "hausdorff") {
                    let dist = hausdorffDistance(queryVectors, group.vectors.toArray())
                    results.add(SearchResult(dist, k, group, None))
                } else {
                    var minDistSum = 0.0
                    for (qv in queryVectors) {
                        var groupMin = Float64.Max
                        for (gv in group.vectors) {
                            let d = distFunc(qv, gv)
                            if (d < groupMin) { groupMin = d }
                        }
                        minDistSum += groupMin
                    }
                    results.add(SearchResult(minDistSum, k, group, None))
                }
            }
        }
        
        sort(results, by: {a, b => a.compare(b)})
        let limit = if (topK < results.size) { topK } else { results.size }
        return Array<SearchResult>(limit, {i => results[i]})
    }

    public func searchWithRepVec(
        queryVector: Array<Float64>,
        topK!: Int64 = 5,
        groupExpansionFactor!: Int64 = 1,
        distanceMethod!: String = "cosine",
        groupName!: Option<String> = None,
        keyOpt!: Option<String> = None,
        vectorType!: Option<String> = None,
        groupType!: Option<String> = None
    ): Array<SearchResult> {
        let groups = search(
            queryVector,
            groupName: groupName,
            topK: topK * groupExpansionFactor,
            mode: "representative",
            distanceMethod: distanceMethod,
            keyOpt: keyOpt,
            vectorType: vectorType,
            groupType: groupType
        )
        let distFunc = _distanceFunc(distanceMethod)
        let candidates = ArrayList<SearchResult>()
        for (gr in groups) {
            for (i in 0..gr.group.vectors.size) {
                candidates.add(SearchResult(distFunc(queryVector, gr.group.vectors[i]), gr.key, gr.group, Some(i)))
            }
        }
        sort(candidates, by: {a, b => a.compare(b)})
        let limit = if (topK < candidates.size) { topK } else { candidates.size }
        return Array<SearchResult>(limit, { i => candidates[i] })
    }

    private func _semanticDistance(queryKeyVector: Option<Array<Float64>>, key: String, distanceMethod: String): Option<Float64> {
        if (queryKeyVector.isNone()) { return None }
        let keyVec = getKeyVector(key)
        if (keyVec.isNone()) { return None }
        let distFunc = _distanceFunc(distanceMethod)
        return Some(distFunc(queryKeyVector.getOrThrow(), keyVec.getOrThrow()))
    }

    public func searchWithRepresentativeRerank(
        queryVector: Array<Float64>,
        queryKeyVector!: Option<Array<Float64>> = None,
        beta!: Float64 = 0.0,
        topK!: Int64 = 5,
        numGroups!: Option<Int64> = None,
        distanceMethod!: String = "cosine"
    ): Array<SearchResult> {
        let groupLimit = match (numGroups) { case Some(n) => n; case None => topK }
        let distFunc = _distanceFunc(distanceMethod)
        let groupScores = ArrayList<SearchResult>()

        for ((key, entry) in this.data) {
            for (group in entry.groups) {
                let repD = distFunc(queryVector, group.representative)
                let semOpt: Option<Float64> = if (beta != 0.0) { _semanticDistance(queryKeyVector, key, distanceMethod) } else { None }
                let mixed = match (semOpt) {
                    case Some(semD) => beta * semD + (1.0 - beta) * repD
                    case None => repD
                }
                groupScores.add(SearchResult(mixed, key, group, None))
            }
        }
        sort(groupScores, by: {a, b => a.compare(b)})

        let candidates = ArrayList<SearchResult>()
        let selected = if (groupLimit < groupScores.size) { groupLimit } else { groupScores.size }
        for (gi in 0..selected) {
            let gr = groupScores[gi]
            let semOpt: Option<Float64> = if (beta != 0.0) { _semanticDistance(queryKeyVector, gr.key, distanceMethod) } else { None }
            for (i in 0..gr.group.vectors.size) {
                let embD = distFunc(queryVector, gr.group.vectors[i])
                let mixed = match (semOpt) {
                    case Some(semD) => beta * semD + (1.0 - beta) * embD
                    case None => embD
                }
                candidates.add(SearchResult(mixed, gr.key, gr.group, Some(i)))
            }
        }
        sort(candidates, by: {a, b => a.compare(b)})
        let limit = if (topK < candidates.size) { topK } else { candidates.size }
        return Array<SearchResult>(limit, { i => candidates[i] })
    }

    public func searchWithMixedKeyRepVec(
        queryVector: Array<Float64>,
        queryKeyVector!: Option<Array<Float64>> = None,
        beta!: Float64 = 0.0,
        topK!: Int64 = 5,
        groupExpansionFactor!: Int64 = 1,
        distanceMethod!: String = "cosine"
    ): Array<SearchResult> {
        if (queryKeyVector.isNone() || beta == 0.0 || this.keyVectors.isEmpty()) {
            return searchWithRepVec(queryVector, topK: topK, groupExpansionFactor: groupExpansionFactor, distanceMethod: distanceMethod)
        }
        let distFunc = _distanceFunc(distanceMethod)
        let groupScores = ArrayList<SearchResult>()
        for ((key, entry) in this.data) {
            for (group in entry.groups) {
                let repD = distFunc(queryVector, group.representative)
                let semOpt = _semanticDistance(queryKeyVector, key, distanceMethod)
                let mixed = match (semOpt) {
                    case Some(semD) => beta * semD + (1.0 - beta) * repD
                    case None => repD
                }
                groupScores.add(SearchResult(mixed, key, group, None))
            }
        }
        sort(groupScores, by: {a, b => a.compare(b)})
        let selected = if (topK * groupExpansionFactor < groupScores.size) { topK * groupExpansionFactor } else { groupScores.size }
        let candidates = ArrayList<SearchResult>()
        for (gi in 0..selected) {
            let gr = groupScores[gi]
            let semOpt = _semanticDistance(queryKeyVector, gr.key, distanceMethod)
            for (i in 0..gr.group.vectors.size) {
                let embD = distFunc(queryVector, gr.group.vectors[i])
                let mixed = match (semOpt) {
                    case Some(semD) => beta * semD + (1.0 - beta) * embD
                    case None => embD
                }
                candidates.add(SearchResult(mixed, gr.key, gr.group, Some(i)))
            }
        }
        sort(candidates, by: {a, b => a.compare(b)})
        let limit = if (topK < candidates.size) { topK } else { candidates.size }
        return Array<SearchResult>(limit, { i => candidates[i] })
    }

    public func buildHdmg(
        embeddingK!: Int64 = 16,
        semanticIntraK!: Int64 = 4,
        semanticBridgeKeys!: Int64 = 2,
        semanticBridgePerKey!: Int64 = 1,
        useMutualEmbedding!: Bool = true
    ): Unit {
        this.hdmgNodeKeys = ArrayList<String>()
        this.hdmgNodeGroups = ArrayList<VectorGroup>()
        this.hdmgBaseKeys = ArrayList<String>()
        this.hdmgEmbeddingEdges = ArrayList<ArrayList<Int64>>()
        this.hdmgSemanticEdges = ArrayList<ArrayList<Int64>>()
        this.hdmgKeyEntries = HashMap<String, Int64>()
        this.hdmgGlobalEntry = None
        this.hdmgBuilt = false

        for ((key, entry) in this.data) {
            for (group in entry.groups) {
                this.hdmgNodeKeys.add(key)
                this.hdmgNodeGroups.add(group)
                this.hdmgBaseKeys.add(_baseKey(key))
            }
        }

        let n = this.hdmgNodeGroups.size
        if (n == 0) {
            this.hdmgLastSearchStats = HashMap<String, Any>()
            this.hdmgLastSearchStats["mode"] = "empty"
            return
        }

        let distMatrix = Array<Array<Float64>>(n, { _ => Array<Float64>(n, repeat: Float64.Max) })
        for (i in 0..n) {
            distMatrix[i][i] = 0.0
        }
        for (i in 0..n) {
            for (j in (i + 1)..n) {
                let d = cosineDistance(this.hdmgNodeGroups[i].representative, this.hdmgNodeGroups[j].representative)
                distMatrix[i][j] = d
                distMatrix[j][i] = d
            }
        }

        let directedKnn = ArrayList<ArrayList<Int64>>()
        for (i in 0..n) {
            let scores = ArrayList<HdmgScore>()
            for (j in 0..n) {
                if (i != j) { scores.add(HdmgScore(distMatrix[i][j], j)) }
            }
            sort(scores, by: { a, b => a.compare(b) })
            let neighbors = ArrayList<Int64>()
            let limit = if (embeddingK < scores.size) { embeddingK } else { scores.size }
            for (p in 0..limit) { neighbors.add(scores[p].nodeId) }
            directedKnn.add(neighbors)
        }

        for (i in 0..n) {
            let edges = ArrayList<Int64>()
            if (useMutualEmbedding) {
                for (nid in directedKnn[i]) {
                    if (directedKnn[nid].contains(i)) { edges.add(nid) }
                }
                for (nid in directedKnn[i]) {
                    if (!edges.contains(nid) && edges.size < embeddingK) { edges.add(nid) }
                }
            } else {
                for (nid in directedKnn[i]) { edges.add(nid) }
            }
            this.hdmgEmbeddingEdges.add(edges)
        }

        let baseKeyToNodeIds = HashMap<String, ArrayList<Int64>>()
        for (i in 0..n) {
            let bk = this.hdmgBaseKeys[i]
            if (!baseKeyToNodeIds.contains(bk)) { baseKeyToNodeIds[bk] = ArrayList<Int64>() }
            baseKeyToNodeIds[bk].add(i)
        }

        for ((bk, nodeIds) in baseKeyToNodeIds) {
            if (nodeIds.size == 1) {
                this.hdmgKeyEntries[bk] = nodeIds[0]
            } else {
                let reps = Array<Array<Float64>>(nodeIds.size, { idx => this.hdmgNodeGroups[nodeIds[idx]].representative })
                let centroid = _meanVector(reps)
                var bestId = nodeIds[0]
                var bestD = cosineDistance(centroid, this.hdmgNodeGroups[bestId].representative)
                for (nid in nodeIds) {
                    let d = cosineDistance(centroid, this.hdmgNodeGroups[nid].representative)
                    if (d < bestD) {
                        bestD = d
                        bestId = nid
                    }
                }
                this.hdmgKeyEntries[bk] = bestId
            }
        }

        let baseKeys = ArrayList<String>()
        for ((bk, _) in baseKeyToNodeIds) { baseKeys.add(bk) }

        for (i in 0..n) {
            let neighbors = ArrayList<Int64>()
            let bk = this.hdmgBaseKeys[i]
            if (baseKeyToNodeIds.contains(bk)) {
                let sameScores = ArrayList<HdmgScore>()
                for (nid in baseKeyToNodeIds[bk]) {
                    if (nid != i) { sameScores.add(HdmgScore(distMatrix[i][nid], nid)) }
                }
                sort(sameScores, by: { a, b => a.compare(b) })
                let sameLimit = if (semanticIntraK < sameScores.size) { semanticIntraK } else { sameScores.size }
                for (p in 0..sameLimit) {
                    if (!neighbors.contains(sameScores[p].nodeId)) { neighbors.add(sameScores[p].nodeId) }
                }
            }

            let qkOpt = getKeyVector(bk)
            if (qkOpt.isSome()) {
                let keyScores = ArrayList<HdmgScore>()
                for (bi in 0..baseKeys.size) {
                    let otherBk = baseKeys[bi]
                    if (otherBk == bk) { continue }
                    let okOpt = getKeyVector(otherBk)
                    if (okOpt.isSome()) {
                        keyScores.add(HdmgScore(cosineDistance(qkOpt.getOrThrow(), okOpt.getOrThrow()), bi))
                    }
                }
                sort(keyScores, by: { a, b => a.compare(b) })
                let bridgeKeyLimit = if (semanticBridgeKeys < keyScores.size) { semanticBridgeKeys } else { keyScores.size }
                for (p in 0..bridgeKeyLimit) {
                    let otherBk = baseKeys[keyScores[p].nodeId]
                    let cands = baseKeyToNodeIds[otherBk]
                    let candScores = ArrayList<HdmgScore>()
                    for (nid in cands) { candScores.add(HdmgScore(distMatrix[i][nid], nid)) }
                    sort(candScores, by: { a, b => a.compare(b) })
                    let candLimit = if (semanticBridgePerKey < candScores.size) { semanticBridgePerKey } else { candScores.size }
                    for (ci in 0..candLimit) {
                        let nid = candScores[ci].nodeId
                        if (!neighbors.contains(nid)) { neighbors.add(nid) }
                    }
                }
            }
            this.hdmgSemanticEdges.add(neighbors)
        }

        let allReps = Array<Array<Float64>>(n, { i => this.hdmgNodeGroups[i].representative })
        let meanRep = _meanVector(allReps)
        var entry = 0
        var entryD = cosineDistance(meanRep, this.hdmgNodeGroups[0].representative)
        for (i in 1..n) {
            let d = cosineDistance(meanRep, this.hdmgNodeGroups[i].representative)
            if (d < entryD) {
                entryD = d
                entry = i
            }
        }
        this.hdmgGlobalEntry = Some(entry)
        this.hdmgBuilt = true
        this.hdmgLastSearchStats = HashMap<String, Any>()
        this.hdmgLastSearchStats["mode"] = "built"
        this.hdmgLastSearchStats["nodes"] = n
        this.hdmgLastSearchStats["embedding_k"] = embeddingK
        this.hdmgLastSearchStats["semantic_intra_k"] = semanticIntraK
        this.hdmgLastSearchStats["semantic_bridge_keys"] = semanticBridgeKeys
        this.hdmgLastSearchStats["semantic_bridge_per_key"] = semanticBridgePerKey
        this.hdmgLastSearchStats["use_mutual_embedding"] = useMutualEmbedding
    }

    public func getLastHdmgSearchStats(): HashMap<String, Any> {
        return this.hdmgLastSearchStats
    }

    public func searchHdmg(
        queryVector: Array<Float64>,
        queryKeyVector!: Option<Array<Float64>> = None,
        alpha!: Float64 = 0.5,
        topK!: Int64 = 10,
        maxSteps!: Int64 = 100,
        distanceMethod!: String = "cosine",
        entryAlphaThreshold!: Float64 = 0.0,
        clusterPoolSize!: Option<Int64> = None,
        topKeyCandidates!: Int64 = 5,
        extraHops!: Int64 = 0
    ): Array<SearchResult> {
        let searchStartNs = _nowNs()
        let pool = match (clusterPoolSize) { case Some(v) => v; case None => topK }
        if (!this.hdmgBuilt || this.hdmgNodeGroups.isEmpty()) {
            let fallback = searchWithMixedKeyRepVec(
                queryVector,
                queryKeyVector: queryKeyVector,
                beta: alpha,
                topK: topK,
                groupExpansionFactor: if (pool > topK) { pool } else { topK },
                distanceMethod: distanceMethod
            )
            this.hdmgLastSearchStats = HashMap<String, Any>()
            this.hdmgLastSearchStats["mode"] = "fallback"
            this.hdmgLastSearchStats["reason"] = "hdmg_not_built"
            this.hdmgLastSearchStats["candidate_pool_size"] = 0
            this.hdmgLastSearchStats["selected_nodes"] = fallback.size
            this.hdmgLastSearchStats["t_total_ns"] = _nowNs() - searchStartNs
            return fallback
        }

        let distFunc = _distanceFunc(distanceMethod)
        var mixedScoreCalls: Int64 = 0
        func mixedScore(nodeId: Int64): Float64 {
            mixedScoreCalls++
            let key = this.hdmgNodeKeys[nodeId]
            let repD = distFunc(queryVector, this.hdmgNodeGroups[nodeId].representative)
            let semOpt = if (queryKeyVector.isSome() && alpha != 0.0) {
                _semanticDistance(queryKeyVector, key, distanceMethod)
            } else { Option<Float64>.None }
            return match (semOpt) {
                case Some(semD) => alpha * semD + (1.0 - alpha) * repD
                case None => repD
            }
        }

        let entryStartNs = _nowNs()
        let entries = ArrayList<Int64>()
        if (queryKeyVector.isSome() && alpha > entryAlphaThreshold && !this.keyVectors.isEmpty()) {
            let keyScores = ArrayList<HdmgScore>()
            for ((bk, nodeId) in this.hdmgKeyEntries) {
                let semOpt = _semanticDistance(queryKeyVector, bk, distanceMethod)
                match (semOpt) {
                    case Some(d) => keyScores.add(HdmgScore(d, nodeId))
                    case None => ()
                }
            }
            sort(keyScores, by: { a, b => a.compare(b) })
            let limit = if (topKeyCandidates < keyScores.size) { topKeyCandidates } else { keyScores.size }
            for (i in 0..limit) {
                if (!entries.contains(keyScores[i].nodeId)) { entries.add(keyScores[i].nodeId) }
            }
        } else {
            let repScores = ArrayList<HdmgScore>()
            for ((_, nodeId) in this.hdmgKeyEntries) {
                repScores.add(HdmgScore(distFunc(queryVector, this.hdmgNodeGroups[nodeId].representative), nodeId))
            }
            sort(repScores, by: { a, b => a.compare(b) })
            let limit = if (topKeyCandidates < repScores.size) { topKeyCandidates } else { repScores.size }
            for (i in 0..limit) {
                if (!entries.contains(repScores[i].nodeId)) { entries.add(repScores[i].nodeId) }
            }
        }
        if (entries.isEmpty() && this.hdmgGlobalEntry.isSome()) { entries.add(this.hdmgGlobalEntry.getOrThrow()) }
        let entryEndNs = _nowNs()

        let candidateNodes = HashMap<Int64, Float64>()
        let scoreCache = HashMap<Int64, Float64>()
        let routeDepths = ArrayList<Int64>()
        let routeHops = ArrayList<Int64>()
        let routeStopReasons = ArrayList<String>()

        func scoreCached(nodeId: Int64): Float64 {
            if (scoreCache.contains(nodeId)) { return scoreCache[nodeId] }
            let s = mixedScore(nodeId)
            scoreCache[nodeId] = s
            return s
        }
        func batchMixedScore(nodeIds: ArrayList<Int64>): HashMap<Int64, Float64> {
            let out = HashMap<Int64, Float64>()
            for (nodeId in nodeIds) {
                out[nodeId] = scoreCached(nodeId)
            }
            return out
        }
        func updateCandidate(nodeId: Int64, score: Float64): Unit {
            if (!candidateNodes.contains(nodeId) || score < candidateNodes[nodeId]) {
                candidateNodes[nodeId] = score
            }
        }

        let walkStartNs = _nowNs()
        for (entryNode in entries) {
            var current = entryNode
            var currentScore = scoreCached(current)
            updateCandidate(current, currentScore)
            var steps: Int64 = 0
            var hops: Int64 = 0
            var stopReason = "max_steps"
            while (steps < maxSteps) {
                steps++
                let allNeighbors = ArrayList<Int64>()
                for (nid in this.hdmgEmbeddingEdges[current]) {
                    if (!allNeighbors.contains(nid)) { allNeighbors.add(nid) }
                }
                for (nid in this.hdmgSemanticEdges[current]) {
                    if (!allNeighbors.contains(nid)) { allNeighbors.add(nid) }
                }
                var nextBest: Option<Int64> = None
                var nextBestScore = currentScore
                let scored = batchMixedScore(allNeighbors)
                for ((nid, s) in scored) {
                    updateCandidate(nid, s)
                    if (s < nextBestScore) {
                        nextBestScore = s
                        nextBest = Some(nid)
                    }
                }
                if (nextBest.isNone()) {
                    stopReason = "no_better_neighbor"
                    break
                }
                current = nextBest.getOrThrow()
                currentScore = nextBestScore
                hops++
            }
            routeDepths.add(steps)
            routeHops.add(hops)
            routeStopReasons.add(stopReason)
        }

        for (_ in 0..extraHops) {
            let frontier = ArrayList<Int64>()
            for ((nid, _) in candidateNodes) { frontier.add(nid) }
            let newNeighbors = ArrayList<Int64>()
            for (nid in frontier) {
                for (nb in this.hdmgEmbeddingEdges[nid]) {
                    if (!candidateNodes.contains(nb) && !newNeighbors.contains(nb)) { newNeighbors.add(nb) }
                }
                for (nb in this.hdmgSemanticEdges[nid]) {
                    if (!candidateNodes.contains(nb) && !newNeighbors.contains(nb)) { newNeighbors.add(nb) }
                }
            }
            let scored = batchMixedScore(newNeighbors)
            for ((nb, s) in scored) { updateCandidate(nb, s) }
        }
        let walkEndNs = _nowNs()

        let rerankStartNs = _nowNs()
        let selectedScores = ArrayList<HdmgScore>()
        for ((nid, s) in candidateNodes) { selectedScores.add(HdmgScore(s, nid)) }
        sort(selectedScores, by: { a, b => a.compare(b) })
        let selectedLimit = if (pool < selectedScores.size) { pool } else { selectedScores.size }

        let candidates = ArrayList<SearchResult>()
        for (si in 0..selectedLimit) {
            let nodeId = selectedScores[si].nodeId
            let key = this.hdmgNodeKeys[nodeId]
            let group = this.hdmgNodeGroups[nodeId]
            let semOpt: Option<Float64> = if (queryKeyVector.isSome() && alpha != 0.0) {
                _semanticDistance(queryKeyVector, key, distanceMethod)
            } else { None }
            for (vi in 0..group.vectors.size) {
                let embD = distFunc(queryVector, group.vectors[vi])
                let score = match (semOpt) {
                    case Some(semD) => alpha * semD + (1.0 - alpha) * embD
                    case None => embD
                }
                candidates.add(SearchResult(score, key, group, Some(vi)))
            }
        }
        sort(candidates, by: { a, b => a.compare(b) })
        let resultLimit = if (topK < candidates.size) { topK } else { candidates.size }
        let results = Array<SearchResult>(resultLimit, { i => candidates[i] })
        let rerankEndNs = _nowNs()
        this.hdmgLastSearchStats = HashMap<String, Any>()
        this.hdmgLastSearchStats["mode"] = "hdmg"
        this.hdmgLastSearchStats["entries_count"] = entries.size
        this.hdmgLastSearchStats["entry_node_ids"] = entries
        this.hdmgLastSearchStats["candidate_pool_size"] = candidateNodes.size
        this.hdmgLastSearchStats["selected_nodes"] = selectedLimit
        this.hdmgLastSearchStats["cluster_pool_size_limit"] = pool
        this.hdmgLastSearchStats["mixed_score_calls"] = mixedScoreCalls
        this.hdmgLastSearchStats["route_depths"] = routeDepths
        this.hdmgLastSearchStats["route_hops"] = routeHops
        this.hdmgLastSearchStats["route_stop_reasons"] = routeStopReasons
        this.hdmgLastSearchStats["max_steps"] = maxSteps
        this.hdmgLastSearchStats["top_key_candidates"] = topKeyCandidates
        this.hdmgLastSearchStats["extra_hops"] = extraHops
        this.hdmgLastSearchStats["entry_alpha_threshold"] = entryAlphaThreshold
        this.hdmgLastSearchStats["t_entry_ns"] = entryEndNs - entryStartNs
        this.hdmgLastSearchStats["t_walk_ns"] = walkEndNs - walkStartNs
        this.hdmgLastSearchStats["t_rerank_ns"] = rerankEndNs - rerankStartNs
        this.hdmgLastSearchStats["t_total_ns"] = rerankEndNs - searchStartNs
        return results
    }
    
    // 获取上下文向量
    public func getContextualVectors(result: SearchResult, num!: Int64 = 3): Array<SearchResult> {
        if (result.vectorIdx.isNone()) { return [result] }
        let vIdx = result.vectorIdx.getOrThrow()
        
        let descMap = result.group.descriptions[vIdx]
        let curIdOpt = descMap["id"] as String
        if (curIdOpt.isNone()) { return [result] }
        let curId = curIdOpt.getOrThrow()

        let lastScoreIdxOpt = curId.lastIndexOf("_")
        if (lastScoreIdxOpt.isNone()) { return [result] }
        let lastScoreIdx = lastScoreIdxOpt.getOrThrow()
        
        let idPrefix = curId[0..lastScoreIdx]
        let idIdxStr = curId[lastScoreIdx + 1..curId.size]
        
        var idIdx = 0
        try { idIdx = Int64.parse(idIdxStr) } catch (e: Exception) { return [result] }

        // 使用原生运算符替代 math.max 消除 warning
        let startIdx = if (0 > idIdx - num) { 0 } else { idIdx - num }
        let endIdx = idIdx + num + 1
        
        let contextualResults = ArrayList<SearchResult>() 

        for (i in startIdx..endIdx) {
            let targetId = "${idPrefix}_${i}"
            if (!this.contextMap.contains(targetId)) { continue }
            
            let mappedValue = this.contextMap[targetId]
            let parts = mappedValue.split("%")
            if (parts.size != 3) { continue }

            let mappedKey = parts[0]
            let mappedGroupName = parts[1]
            let mappedInsertIdx = Int64.parse(parts[2])

            let mappedGroupOpt = getGroupByName(mappedKey, mappedGroupName)
            if (mappedGroupOpt.isNone()) { continue }
            let mappedGroup = mappedGroupOpt.getOrThrow()

            if (mappedInsertIdx >= 0 && mappedInsertIdx < mappedGroup.size()) {
                contextualResults.add(SearchResult(result.distance, mappedKey, mappedGroup, Some(mappedInsertIdx)))
            }
        }

        if (contextualResults.isEmpty()) { return [result] }
        return Array<SearchResult>(contextualResults.size, {i => contextualResults[i]})
    }

    // 获取关联的配对向量
    public func getPairedVectors(key: String, groupId: String): ArrayList<(VectorGroup, HashMap<String, Any>)> {
        let pairedGroups = ArrayList<(VectorGroup, HashMap<String, Any>)>()
        if (!this.data.contains(key)) { return pairedGroups }
        
        let relations = getRelations(key)
        let pairListOpt = relations["pair"] as ArrayList<HashMap<String, Any>>
        if (pairListOpt.isNone()) { return pairedGroups }
        
        for (pairRel in pairListOpt.getOrThrow()) {
            var otherGroupId: Option<String> = None
            let g1id = (pairRel["group1_id"] as String).getOrThrow()
            let g2id = (pairRel["group2_id"] as String).getOrThrow()
            
            if (g1id == groupId) { otherGroupId = Some(g2id) }
            else if (g2id == groupId) { otherGroupId = Some(g1id) }
            
            if (otherGroupId.isSome()) {
                let otherGroupOpt = getGroupById(key, otherGroupId.getOrThrow())
                if (otherGroupOpt.isSome()) {
                    pairedGroups.add((otherGroupOpt.getOrThrow(), pairRel))
                }
            }
        }
        return pairedGroups
    }

    // ============= 统计信息与分析 =============
    
    public func getStatistics(): HashMap<String, Any> {
        let stats = HashMap<String, Any>()
        stats["total_keys"] = this.data.size
        var totalGroups = 0
        var totalVectors = 0
        
        let vectorTypes = ArrayList<String>()
        let groupTypes = ArrayList<String>()
        var totalPairRelations = 0
        var totalTreeRelations = 0
        let keyDetails = HashMap<String, Any>()

        for ((key, entry) in this.data) {
            totalGroups += entry.groups.size
            var keyVectors = 0
            for (group in entry.groups) {
                keyVectors += Int64(group.size())
                if (!vectorTypes.contains(group.vectorType)) { vectorTypes.add(group.vectorType) }
                if (!groupTypes.contains(group.groupType)) { groupTypes.add(group.groupType) }
            }
            totalVectors += keyVectors

            let relations = entry.metadata["relations"] as HashMap<String, Any>
            var pCount = 0
            var tCount = 0
            if (relations.isSome()) {
                let pList = relations.getOrThrow()["pair"] as ArrayList<HashMap<String, Any>>
                let tList = relations.getOrThrow()["tree"] as ArrayList<HashMap<String, Any>>
                pCount = if (pList.isSome()) { pList.getOrThrow().size } else { 0 }
                tCount = if (tList.isSome()) { tList.getOrThrow().size } else { 0 }
                totalPairRelations += pCount
                totalTreeRelations += tCount
            }

            let details = HashMap<String, Any>()
            details["groups_count"] = entry.groups.size
            details["vectors_count"] = keyVectors
            details["pair_relations"] = pCount
            details["tree_relations"] = tCount
            
            let vTypesOpt = entry.metadata["vector_types"] as ArrayList<String>
            let gTypesOpt = entry.metadata["group_types"] as ArrayList<String>
            
            let vList = ArrayList<String>()
            if (vTypesOpt.isSome()) { for (v in vTypesOpt.getOrThrow()) { vList.add(v) } }
            details["vector_types"] = vList
            
            let gList = ArrayList<String>()
            if (gTypesOpt.isSome()) { for (g in gTypesOpt.getOrThrow()) { gList.add(g) } }
            details["group_types"] = gList
            
            keyDetails[key] = details
        }

        stats["total_groups"] = totalGroups
        stats["total_vectors"] = totalVectors
        stats["vector_types"] = vectorTypes
        stats["group_types"] = groupTypes
        stats["total_pair_relations"] = totalPairRelations
        stats["total_tree_relations"] = totalTreeRelations
        stats["key_details"] = keyDetails

        return stats
    }

    public func analyzeRelationships(keyOpt!: Option<String> = None): HashMap<String, Any> {
        let analysis = HashMap<String, Any>()
        
        let pairAnalysis = HashMap<String, Any>()
        var totalPairs = 0
        let relationTypes = ArrayList<String>()
        pairAnalysis["avg_vector_count"] = 0
        
        let treeAnalysis = HashMap<String, Any>()
        var maxDepth = 0
        let treeRootNodes = ArrayList<String>()
        
        let connectivity = HashMap<String, Any>()
        let connectedGroups = ArrayList<String>()
        let isolatedGroups = ArrayList<String>()

        let searchKeys = match(keyOpt) { case Some(k) => [k]; case None => _getKeysAsArray() }

        func calculateDepth(node: String, pChildren: HashMap<String, ArrayList<String>>, currentDepth: Int64): Int64 {
            if (!pChildren.contains(node) || pChildren[node].isEmpty()) { return currentDepth }
            var maxChildDepth = currentDepth
            for (child in pChildren[node]) {
                let d = calculateDepth(child, pChildren, currentDepth + 1)
                if (d > maxChildDepth) { maxChildDepth = d }
            }
            return maxChildDepth
        }

        for (k in searchKeys) {
            if (!this.data.contains(k)) { continue }
            let relations = getRelations(k)
            
            let pairListOpt = relations["pair"] as ArrayList<HashMap<String, Any>>
            if (pairListOpt.isSome()) {
                for (pairRel in pairListOpt.getOrThrow()) {
                    totalPairs++
                    
                    let rTypeOpt = pairRel["relation_type"] as String
                    let rType = match (rTypeOpt) {
                        case Some(s) => s
                        case None => "unknown"
                    }
                    
                    if (!relationTypes.contains(rType)) { relationTypes.add(rType) }
                    
                    let grp1 = (pairRel["group1_id"] as String).getOrThrow()
                    let grp2 = (pairRel["group2_id"] as String).getOrThrow()
                    if (!connectedGroups.contains(grp1)) { connectedGroups.add(grp1) }
                    if (!connectedGroups.contains(grp2)) { connectedGroups.add(grp2) }
                }
            }

            let hierarchy = getHierarchyTree(k)
            let rootsOpt = hierarchy["root_nodes"] as ArrayList<String>
            let pChildrenOpt = hierarchy["parent_children"] as HashMap<String, ArrayList<String>>
            
            if (rootsOpt.isSome() && pChildrenOpt.isSome()) {
                let roots = rootsOpt.getOrThrow()
                let pChildren = pChildrenOpt.getOrThrow()
                for (r in roots) { 
                    if (!treeRootNodes.contains(r)) { treeRootNodes.add(r) }
                }
                
                for (root in roots) {
                    let depth = calculateDepth(root, pChildren, 0)
                    if (depth > maxDepth) { maxDepth = depth }
                }
            }
        }

        pairAnalysis["total_pairs"] = totalPairs
        pairAnalysis["relation_types"] = relationTypes
        treeAnalysis["max_depth"] = maxDepth
        treeAnalysis["root_nodes"] = treeRootNodes
        connectivity["connected_groups"] = connectedGroups
        connectivity["isolated_groups"] = isolatedGroups

        analysis["pair_analysis"] = pairAnalysis
        analysis["tree_analysis"] = treeAnalysis
        analysis["connectivity"] = connectivity

        return analysis
    }

    public func printHierarchyTree(key: String, showGroupInfo!: Bool = true): Unit {
        let hierarchy = getHierarchyTree(key)
        let rootNodesOpt = hierarchy["root_nodes"] as ArrayList<String>
        if (rootNodesOpt.isNone() || rootNodesOpt.getOrThrow().isEmpty()) {
            println("没有找到层次结构数据")
            return
        }
        
        println("--- ${key} 的层次结构树形图 ---")
        let parentChildren = (hierarchy["parent_children"] as HashMap<String, ArrayList<String>>).getOrThrow()
        let nodeGroups = if (showGroupInfo) { (hierarchy["node_groups"] as HashMap<String, ArrayList<String>>).getOrThrow() } else { HashMap<String, ArrayList<String>>() }
        
        _printTreeStructure(parentChildren, rootNodesOpt.getOrThrow(), nodeGroups, showGroupInfo, "")
    }

    private func _printTreeStructure(
        parentChildren: HashMap<String, ArrayList<String>>, rootNodes: ArrayList<String>, 
        nodeGroups: HashMap<String, ArrayList<String>>, showGroupInfo: Bool, 
        prefix: String
    ): Unit {
        for (i in 0..rootNodes.size) {
            let root = rootNodes[i]
            let isLastNode = (i == rootNodes.size - 1)
            let currentPrefix = prefix + (if (isLastNode) { "└── " } else { "├── " })
            
            var groupInfo = ""
            if (showGroupInfo && nodeGroups.contains(root)) {
                let groupIds = nodeGroups[root]
                groupInfo = if (groupIds.size > 0) { " (向量组个数: ${groupIds.size})" } else { " (无向量组)" }
            }
            
            println("${currentPrefix}${root}${groupInfo}")
            
            if (parentChildren.contains(root)) {
                let children = parentChildren[root]
                if (!children.isEmpty()) {
                    let childPrefix = prefix + (if (isLastNode) { "    " } else { "│   " })
                    _printTreeStructure(parentChildren, children, nodeGroups, showGroupInfo, childPrefix)
                }
            }
        }
    }

    public func size(): Int64 {
        return this.data.size
    }
    
    public func contains(key: String): Bool {
        return this.data.contains(key)
    }

    public operator func [](key: String): VectorMapEntry {
        return this.data[key]
    }

    public func toString(): String {
        var groupCount = 0
        for ((_, entry) in this.data) { groupCount += entry.groups.size }
        return "VectorMap(keys=${this.data.size}, groups=${groupCount})"
    }
}
