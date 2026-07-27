package semanticvg_cj.examples

import std.collection.*
import std.convert.*
import std.fs.*
import semanticvg_cj.storage.*

private class CodeEntityInput {
    let id: String
    let key: String
    let aspect: String
    let modality: String
    let path: String
    let line: String
    let kind: String
    let text: String
    let vector: Array<Float64>

    init(id: String, key: String, aspect: String, modality: String, path: String,
         line: String, kind: String, text: String, vector: Array<Float64>) {
        this.id = id
        this.key = key
        this.aspect = aspect
        this.modality = modality
        this.path = path
        this.line = line
        this.kind = kind
        this.text = text
        this.vector = vector
    }
}

private class CodeKeyInput {
    let id: String
    let text: String
    let vector: Array<Float64>

    init(id: String, text: String, vector: Array<Float64>) {
        this.id = id
        this.text = text
        this.vector = vector
    }
}

private class CodeRelationInput {
    let key: String
    let source: String
    let target: String
    let relationType: String

    init(key: String, source: String, target: String, relationType: String) {
        this.key = key
        this.source = source
        this.target = target
        this.relationType = relationType
    }
}

private class CodeQueryInput {
    let id: String
    let paradigm: String
    let text: String
    let targetKey: String
    let seed: String
    let relationType: String
    let gold: ArrayList<String>
    let aspects: ArrayList<String>
    let vector: Array<Float64>

    init(id: String, paradigm: String, text: String, targetKey: String, seed: String,
         relationType: String, gold: ArrayList<String>, aspects: ArrayList<String>,
         vector: Array<Float64>) {
        this.id = id
        this.paradigm = paradigm
        this.text = text
        this.targetKey = targetKey
        this.seed = seed
        this.relationType = relationType
        this.gold = gold
        this.aspects = aspects
        this.vector = vector
    }
}

private class CodeCaseInput {
    var protocol: String = ""
    var dimension: Int64 = 0
    let entities = ArrayList<CodeEntityInput>()
    let keys = ArrayList<CodeKeyInput>()
    let relations = ArrayList<CodeRelationInput>()
    let queries = ArrayList<CodeQueryInput>()
}

private func parseCodeVector(raw: String): Array<Float64> {
    let parts = raw.split(",")
    return Array<Float64>(parts.size, { i => Float64.parse(parts[i]) })
}

private func parseCodeIds(raw: String): ArrayList<String> {
    let ids = ArrayList<String>()
    if (!raw.isEmpty() && raw != "-") {
        for (item in raw.split(",")) { ids.add(item) }
    }
    return ids
}

private func parseCodeCaseLine(raw: String, input: CodeCaseInput): Unit {
    let line = raw.replace("\r", "")
    if (line.isEmpty()) { return }
    let parts = line.split("\t")
    if (parts[0] == "META") {
        if (parts.size != 3) { throw IllegalArgumentException("invalid code case META line") }
        input.protocol = parts[1]
        input.dimension = Int64.parse(parts[2])
    } else if (parts[0] == "ENTITY") {
        if (parts.size != 10) { throw IllegalArgumentException("invalid code case ENTITY line") }
        input.entities.add(CodeEntityInput(
            parts[1], parts[2], parts[3], parts[4], parts[5], parts[6], parts[7],
            parts[8], parseCodeVector(parts[9])
        ))
    } else if (parts[0] == "KEY") {
        if (parts.size != 4) { throw IllegalArgumentException("invalid code case KEY line") }
        input.keys.add(CodeKeyInput(parts[1], parts[2], parseCodeVector(parts[3])))
    } else if (parts[0] == "REL") {
        if (parts.size != 5) { throw IllegalArgumentException("invalid code case REL line") }
        input.relations.add(CodeRelationInput(parts[1], parts[2], parts[3], parts[4]))
    } else if (parts[0] == "QUERY") {
        if (parts.size != 10) { throw IllegalArgumentException("invalid code case QUERY line") }
        input.queries.add(CodeQueryInput(
            parts[1], parts[2], parts[3], parts[4], parts[5], parts[6],
            parseCodeIds(parts[7]), parseCodeIds(parts[8]), parseCodeVector(parts[9])
        ))
    } else {
        throw IllegalArgumentException("unknown code case command: ${parts[0]}")
    }
}

private func loadCodeCaseInput(filePath: String): CodeCaseInput {
    let input = CodeCaseInput()
    let file = File(filePath, OpenMode.Read)
    let buffer = Array<Byte>(65536, repeat: 0)
    var carry = ""
    while (true) {
        let count = file.read(buffer)
        if (count <= 0) { break }
        carry += String.fromUtf8(Array<Byte>(count, { i => buffer[i] }))
        while (true) {
            match (carry.indexOf("\n")) {
                case Some(index) =>
                    parseCodeCaseLine(carry[0..index], input)
                    carry = carry[(index + 1)..carry.size]
                case None => break
            }
        }
    }
    if (!carry.isEmpty()) { parseCodeCaseLine(carry, input) }
    if (input.protocol != "code-context-four-paradigm-v2") {
        throw IllegalArgumentException("invalid code case protocol: ${input.protocol}")
    }
    if (input.entities.isEmpty() || input.keys.isEmpty() || input.queries.isEmpty()) {
        throw IllegalArgumentException("code case input cannot be empty")
    }
    return input
}

private func entityId(group: VectorGroup): String {
    let value = group.descriptions[0]["entity_id"] as String
    return if (value.isSome()) { value.getOrThrow() } else { group.groupName }
}

private func entityAspect(group: VectorGroup): String {
    let value = group.descriptions[0]["aspect"] as String
    return if (value.isSome()) { value.getOrThrow() } else { "unknown" }
}

private func appendUnique(items: ArrayList<String>, item: String, budget: Int64): Bool {
    if (items.size >= budget || items.contains(item)) { return false }
    items.add(item)
    return true
}

private func joined(items: ArrayList<String>): String {
    var output = ""
    for (i in 0..items.size) {
        if (i > 0) { output += "," }
        output += items[i]
    }
    return output
}

private func idsFromResults(results: Array<SearchResult>, budget: Int64): ArrayList<String> {
    let ids = ArrayList<String>()
    for (result in results) { appendUnique(ids, entityId(result.group), budget) }
    return ids
}

private func goldHits(gold: ArrayList<String>, ranked: ArrayList<String>): Int64 {
    var hits: Int64 = 0
    for (id in gold) {
        if (ranked.contains(id)) { hits++ }
    }
    return hits
}

private func printResult(query: CodeQueryInput, method: String, metric: String,
                         ranked: ArrayList<String>, hits: Int64, total: Int64): Unit {
    let score = if (total <= 0) { 0.0 } else { Float64(hits) / Float64(total) }
    println(
        "CODE_CASE|RESULT|${query.id}|${query.paradigm}|${method}|${metric}|" +
        "${joined(ranked)}|${hits}|${total}|${score}"
    )
}

private func runEar(query: CodeQueryInput, vectorMap: VectorMap): Unit {
    let budget: Int64 = 5
    let flatResults = vectorMap.search(query.vector, topK: budget, mode: "single")
    let flatIds = idsFromResults(flatResults, budget)
    var flatHits: Int64 = 0
    for (result in flatResults) {
        if (result.key == query.targetKey) { flatHits++ }
    }

    let alignedResults = vectorMap.searchWithMixedKeyRepVec(
        query.vector,
        queryKeyVector: Some(query.vector),
        beta: 0.75,
        topK: budget,
        groupExpansionFactor: 4
    )
    let alignedIds = idsFromResults(alignedResults, budget)
    var alignedHits: Int64 = 0
    for (result in alignedResults) {
        if (result.key == query.targetKey) { alignedHits++ }
    }
    printResult(query, "flat_vector", "entity_purity_at_5", flatIds, flatHits, budget)
    printResult(query, "violas_ear", "entity_purity_at_5", alignedIds, alignedHits, budget)
}

private func runDdr(query: CodeQueryInput, vectorMap: VectorMap): Unit {
    let budget = query.aspects.size
    let allResults = vectorMap.search(
        query.vector, topK: 64, mode: "single", keyOpt: Some(query.targetKey)
    )
    let flatIds = idsFromResults(allResults, budget)
    let flatAspects = ArrayList<String>()
    for (result in allResults) {
        let id = entityId(result.group)
        if (!flatIds.contains(id)) { continue }
        let aspect = entityAspect(result.group)
        if (query.aspects.contains(aspect) && !flatAspects.contains(aspect)) {
            flatAspects.add(aspect)
        }
    }

    let diverseIds = ArrayList<String>()
    let selectedAspects = ArrayList<String>()
    for (result in allResults) {
        let aspect = entityAspect(result.group)
        if (query.aspects.contains(aspect) && !selectedAspects.contains(aspect)) {
            selectedAspects.add(aspect)
            appendUnique(diverseIds, entityId(result.group), budget)
        }
        if (diverseIds.size >= budget) { break }
    }
    for (id in flatIds) { appendUnique(diverseIds, id, budget) }

    printResult(query, "flat_vector", "aspect_coverage_at_5", flatIds, flatAspects.size, budget)
    printResult(query, "violas_ddr", "aspect_coverage_at_5", diverseIds, selectedAspects.size, budget)
}

private func runRer(query: CodeQueryInput, vectorMap: VectorMap,
                    groups: HashMap<String, VectorGroup>): Unit {
    let budget = query.gold.size
    let flatResults = vectorMap.search(
        query.vector, topK: budget, mode: "single", keyOpt: Some(query.targetKey)
    )
    let flatIds = idsFromResults(flatResults, budget)

    if (!groups.contains(query.seed)) {
        throw IllegalArgumentException("unknown RER seed: ${query.seed}")
    }
    let expanded = ArrayList<String>()
    let frontier = ArrayList<VectorGroup>()
    appendUnique(expanded, query.seed, budget)
    frontier.add(groups[query.seed])
    var cursor: Int64 = 0
    while (cursor < frontier.size && expanded.size < budget) {
        let source = frontier[cursor]
        for ((target, relation) in vectorMap.getPairedVectors(query.targetKey, source.groupId)) {
            let typeOpt = relation["relation_type"] as String
            let relationType = if (typeOpt.isSome()) { typeOpt.getOrThrow() } else { "unknown" }
            if (query.relationType != "*" && query.relationType != relationType) { continue }
            if (appendUnique(expanded, entityId(target), budget)) { frontier.add(target) }
            if (expanded.size >= budget) { break }
        }
        cursor++
    }
    for (id in flatIds) { appendUnique(expanded, id, budget) }

    printResult(query, "flat_vector", "dependency_coverage_at_5", flatIds,
        goldHits(query.gold, flatIds), query.gold.size)
    printResult(query, "violas_rer", "dependency_coverage_at_5", expanded,
        goldHits(query.gold, expanded), query.gold.size)
}

private func runCmp(query: CodeQueryInput, vectorMap: VectorMap,
                    groups: HashMap<String, VectorGroup>): Unit {
    if (!groups.contains(query.seed)) {
        throw IllegalArgumentException("unknown CMP seed: ${query.seed}")
    }
    let seed = groups[query.seed]
    let neighborResults = vectorMap.search(
        seed.vectors[0], topK: 8, mode: "single", keyOpt: Some(query.targetKey)
    )
    let flatIds = ArrayList<String>()
    for (result in neighborResults) {
        let id = entityId(result.group)
        if (id != query.seed) { appendUnique(flatIds, id, query.gold.size) }
    }

    let pairedIds = ArrayList<String>()
    for ((target, relation) in vectorMap.getPairedVectors(query.targetKey, seed.groupId)) {
        let typeOpt = relation["relation_type"] as String
        let relationType = if (typeOpt.isSome()) { typeOpt.getOrThrow() } else { "unknown" }
        if (relationType == query.relationType) {
            appendUnique(pairedIds, entityId(target), query.gold.size)
        }
    }
    printResult(query, "flat_neighbor", "pair_hit_at_1", flatIds,
        goldHits(query.gold, flatIds), query.gold.size)
    printResult(query, "violas_cmp", "pair_hit_at_1", pairedIds,
        goldHits(query.gold, pairedIds), query.gold.size)
}

public func runCodeContextCaseStudy(filePath: String): Unit {
    let input = loadCodeCaseInput(filePath)
    let vectorMap = VectorMap()
    let groups = HashMap<String, VectorGroup>()

    for (entity in input.entities) {
        if (entity.vector.size != input.dimension) {
            throw IllegalArgumentException("entity vector dimension mismatch: ${entity.id}")
        }
        let description = HashMap<String, Any>()
        description["entity_id"] = entity.id
        description["aspect"] = entity.aspect
        description["modality"] = entity.modality
        description["path"] = entity.path
        description["line"] = entity.line
        description["kind"] = entity.kind
        description["text"] = entity.text
        let descriptions = Array<Any>(1, { _ => description })
        let group = VectorGroup(
            entity.id, entity.vector, entity.text, [entity.vector], descriptions,
            vectorType: entity.modality, groupType: entity.aspect
        )
        vectorMap.insert(entity.key, group, metadataOpt: None)
        groups[entity.id] = group
    }

    let keyVectors = HashMap<String, Array<Float64>>()
    for (key in input.keys) {
        if (key.vector.size != input.dimension) {
            throw IllegalArgumentException("key vector dimension mismatch: ${key.id}")
        }
        keyVectors[key.id] = key.vector
    }
    vectorMap.setKeyVectors(keyVectors)

    for (relation in input.relations) {
        if (!groups.contains(relation.source) || !groups.contains(relation.target)) {
            throw IllegalArgumentException("relation references an unknown entity")
        }
        let added = vectorMap.addPairRelation(
            relation.key, groups[relation.source].groupId, groups[relation.target].groupId,
            relationType: relation.relationType
        )
        if (!added) { throw IllegalStateException("failed to add relation: ${relation.relationType}") }
    }

    for (query in input.queries) {
        if (query.vector.size != input.dimension) {
            throw IllegalArgumentException("query vector dimension mismatch: ${query.id}")
        }
        if (query.paradigm == "EAR") {
            runEar(query, vectorMap)
        } else if (query.paradigm == "DDR") {
            runDdr(query, vectorMap)
        } else if (query.paradigm == "RER") {
            runRer(query, vectorMap, groups)
        } else if (query.paradigm == "CMP") {
            runCmp(query, vectorMap, groups)
        } else {
            throw IllegalArgumentException("unsupported code case paradigm: ${query.paradigm}")
        }
    }
}
