#!/usr/bin/env python3
"""End-to-end tests for the OpenClaw semantic memory skill flow."""

import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC_DIR)

from semantic_memory_v2 import create_semantic_memory


TEST_MEMORY_TEMPLATES = [
    {
        "text": "老板喜欢直接、简洁的反馈方式，不喜欢冗长解释。",
        "metadata": {
            "type": "conversation",
            "category": "preference",
            "source": "conversation",
            "session_id": "sess-pref-001",
            "timestamp": "2026-03-13T14:00:00Z",
            "tags": ["communication", "feedback"],
        },
    },
    {
        "text": "老板偏好 bullet points 列表。",
        "metadata": {
            "type": "conversation",
            "category": "preference",
            "source": "conversation",
            "session_id": "sess-pref-002",
            "timestamp": "2026-03-13T14:02:00Z",
            "tags": ["communication", "format"],
        },
    },
    {
        "text": "老板的时区是 America/Los_Angeles。",
        "metadata": {
            "type": "conversation",
            "category": "info",
            "source": "conversation",
            "session_id": "sess-info-001",
            "timestamp": "2026-03-13T14:05:00Z",
            "tags": ["timezone"],
        },
    },
    {
        "text": "Polymarket 监控是日常任务。",
        "metadata": {
            "type": "task",
            "category": "task",
            "source": "system",
            "session_id": "sess-task-001",
            "timestamp": "2026-03-13T14:10:00Z",
            "tags": ["monitoring", "routine"],
        },
    },
    {
        "text": "Semantic Memory 使用 SQLite-vec 作为本地向量数据库。",
        "metadata": {
            "type": "system_note",
            "category": "architecture",
            "source": "system",
            "session_id": "sess-arch-001",
            "timestamp": "2026-03-13T14:20:00Z",
            "tags": ["vector-db", "architecture"],
        },
    },
]


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def assert_result_contains(results, expected_texts, message):
    returned_texts = [item["text"] for item in results]
    assert_true(
        any(text in expected_texts for text in returned_texts),
        f"{message}. returned={returned_texts}",
    )


def build_test_memories(run_tag):
    memories = []
    for index, item in enumerate(TEST_MEMORY_TEMPLATES):
        metadata = dict(item["metadata"])
        tags = list(metadata.get("tags", []))
        tags.append(run_tag)
        metadata["tags"] = tags
        metadata["session_id"] = f"{metadata['session_id']}-{run_tag}-{index}"
        memories.append({"text": item["text"], "metadata": metadata})
    return memories


def filter_results_by_tag(results, run_tag):
    tagged_results = []
    for item in results:
        tags = item.get("metadata", {}).get("tags", [])
        if run_tag in tags:
            tagged_results.append(item)
    return tagged_results


def main():
    print("=" * 72)
    print("OpenClaw Semantic Memory Skill Flow Test")
    print("=" * 72)

    with tempfile.NamedTemporaryFile(prefix="openclaw-memory-", suffix=".db", delete=False) as temp_file:
        db_path = temp_file.name

    run_tag = f"openclaw-test-{uuid.uuid4().hex[:8]}"

    memory = create_semantic_memory(
        grpc_address="localhost:50051",
        db_path=db_path,
    )

    created_ids = []
    try:
        print("\n[1/6] Health check")
        assert_true(memory.health_check(), "gRPC service is not healthy")
        print("  OK: service is healthy")

        baseline_count = memory.count()
        test_memories = build_test_memories(run_tag)

        print("\n[2/6] Insert OpenClaw-style memories")
        for item in test_memories:
            memory_id = memory.add_memory(item["text"], item["metadata"])
            created_ids.append(memory_id)
            print(f"  OK: inserted {memory_id[:8]}... {item['metadata']['category']}")

        assert_true(
            memory.count() == baseline_count + len(test_memories),
            "memory count mismatch after inserts",
        )

        print("\n[3/6] Preference retrieval with rerank")
        preference_results = memory.search("老板喜欢什么样的反馈方式？", top_k=10, use_rerank=True)
        preference_results_tagged = filter_results_by_tag(preference_results, run_tag)
        assert_true(len(preference_results_tagged) >= 2, "expected at least two tagged preference results")
        assert_result_contains(
            preference_results_tagged,
            {
                "老板喜欢直接、简洁的反馈方式，不喜欢冗长解释。",
                "老板偏好 bullet points 列表。",
            },
            "reranked preference results did not include expected tagged memories",
        )
        for item in preference_results:
            print(f"  {item['similarity']:.6f} | {item['metadata']['category']} | {item['text']}")

        print("\n[4/6] Factual retrieval: coarse recall then rerank correction")
        info_results_fast = memory.search("老板的时区是什么？", top_k=10, use_rerank=False)
        info_results_fast_tagged = filter_results_by_tag(info_results_fast, run_tag)
        assert_true(len(info_results_fast) >= 1, "expected at least one fast retrieval result")
        assert_result_contains(
            info_results_fast_tagged,
            {"老板的时区是 America/Los_Angeles。"},
            "coarse retrieval did not include the tagged timezone memory in top_k",
        )

        info_results_rerank = memory.search("老板的时区是什么？", top_k=10, use_rerank=True)
        info_results_rerank_tagged = filter_results_by_tag(info_results_rerank, run_tag)
        assert_true(len(info_results_rerank) >= 1, "expected at least one reranked result")
        assert_true(
            len(info_results_rerank_tagged) >= 1,
            "rerank did not return the tagged timezone memory",
        )
        assert_true(
            info_results_rerank_tagged[0]["text"] == "老板的时区是 America/Los_Angeles。",
            f"rerank failed to prioritize the tagged timezone memory: {info_results_rerank_tagged[0]['text']}",
        )
        assert_true(
            info_results_rerank_tagged[0]["metadata"]["category"] == "info",
            "metadata category was not preserved for timezone memory",
        )
        print(f"  Fast top candidates = {[item['text'] for item in info_results_fast]}")
        print(f"  Tagged rerank top result = {info_results_rerank_tagged[0]['text']}")

        print("\n[5/6] Architecture/task routing queries")
        architecture_results = memory.search("系统用什么向量数据库？", top_k=10, use_rerank=True)
        architecture_results_tagged = filter_results_by_tag(architecture_results, run_tag)
        assert_true(len(architecture_results_tagged) >= 1, "expected at least one tagged architecture result")
        assert_true(
            architecture_results_tagged[0]["text"] == "Semantic Memory 使用 SQLite-vec 作为本地向量数据库。",
            f"tagged architecture result mismatch: {architecture_results_tagged[0]['text']}",
        )

        task_results = memory.search("有什么日常监控任务？", top_k=10, use_rerank=True)
        task_results_tagged = filter_results_by_tag(task_results, run_tag)
        assert_result_contains(
            task_results_tagged,
            {"Polymarket 监控是日常任务。"},
            "task query did not surface the tagged routine monitoring memory",
        )
        print(f"  OK: tagged architecture top = {architecture_results_tagged[0]['text']}")
        print(f"  OK: task top candidates = {[item['text'] for item in task_results]}")

        print("\n[6/6] Delete and cleanup semantics")
        deleted_id = created_ids[0]
        deleted = memory.delete_memory(deleted_id)
        assert_true(deleted, "delete_memory returned False")
        created_ids.pop(0)
        assert_true(
            memory.count() == baseline_count + len(test_memories) - 1,
            "memory count mismatch after delete",
        )
        assert_true(memory.get_memory(deleted_id) is None, "deleted memory still retrievable by id")
        print("  OK: delete removed the target memory")

        print("\nAll OpenClaw skill-flow checks passed.")
        print(f"Executed at: {datetime.now(timezone.utc).isoformat()}")
        return 0
    finally:
        for memory_id in reversed(created_ids):
            try:
                deleted = memory.delete_memory(memory_id)
                if deleted:
                    print(f"  cleanup: removed {memory_id[:8]}...")
            except Exception as exc:
                print(f"  cleanup warning: failed to remove {memory_id}: {exc}")
        memory.close()
        if os.path.exists(db_path):
            os.remove(db_path)


if __name__ == "__main__":
    raise SystemExit(main())