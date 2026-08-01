---
name: project-improvement-review
description: Perform a read-only, evidence-backed review of an entire software repository using the installed codebase MCP and its project knowledge graph. Use when asked to understand a product holistically, inventory implemented capabilities, compare implementation with intended user experience, find incomplete/disconnected/duplicated/underused functionality, or produce prioritized architectural, product, reliability, performance, observability, testing, documentation, and developer-experience improvements. Do not use as a security audit or as a changed-files-only code review.
---

# Project Improvement Review

Analyze the whole product before recommending changes. Use the codebase MCP as the primary discovery mechanism and verify important claims against source and repository documentation. Remain read-only unless the user separately and explicitly requests implementation.

## Non-negotiable rules

- Index the complete repository with the codebase MCP before analysis. Refresh an existing index; do not assume it is current.
- Review product intent, architecture, documentation, entry points, user flows, integrations, storage, tests, operational behavior, and frontend/backend connections—not only changed files.
- Cite exact repository evidence for every finding and recommendation. Prefer file paths plus qualified symbols or knowledge-graph entities.
- Label facts as **Confirmed** only when directly supported by code, configuration, tests, documentation, or graph relationships.
- Label product ideas, likely intent, or expected value as **Inferred opportunity** and state the inference basis.
- Exclude generic advice that cannot be tied to repository evidence.
- Do not turn the review into a vulnerability scan. Mention security only when repository evidence makes it a direct product, reliability, or deployment constraint.
- Do not edit code, configuration, documentation, tickets, or external systems unless the user explicitly requests changes after the review.

## Workflow

### 1. Establish scope and index

1. Resolve the repository root and inspect repository-level instructions.
2. Call the codebase MCP repository indexer in `full` mode for the root.
3. Record the indexed project name, exclusions, node/edge counts, languages, and entry points.
4. If indexing fails, report the limitation; do not pretend to have completed a whole-repository review.

### 2. Reconstruct product intent

Use both narrative and executable evidence:

- Inspect README files, product/architecture documents, examples, setup instructions, package manifests, environment templates, and user-facing copy.
- Use the graph architecture view for structure, dependencies, routes, entry points, hotspots, boundaries, layers, clusters, and file tree.
- Trace primary entry points and user flows through calls, data flow, routes, persistence, and external integrations.
- Compare product claims in documentation with code that is actually connected to runtime entry points.

Summarize intended users and problem solved, primary experience, runtime modes, data/control flow, external systems, operational assumptions, and the difference between current scope and aspirational positioning.

### 3. Build the capability inventory

Inventory capabilities by product area rather than directory alone. For each capability record:

- user outcome;
- implementation status: `connected`, `partial`, `disconnected`, `duplicate`, or `supporting infrastructure`;
- runtime entry point or trigger;
- primary files/modules/symbols;
- upstream/downstream graph relationships;
- validation evidence such as tests, scripts, or observed build/run behavior.

Use graph search for definitions and relationships. Detect pagination with `has_more` and continue or narrow the query. Use code search for configuration, product copy, TODO markers, feature flags, and documentation claims. Use source snippets only after locating the exact qualified symbol.

### 4. Find confirmed gaps

Actively check for:

- documented capabilities with no runtime path;
- modules that exist but are not called from an entry point;
- frontend states with no backend producer, or backend behavior with no user-facing surface;
- duplicate implementations of the same responsibility;
- optional integrations that fail silently or lack an observable state;
- persistence paths that omit important product data;
- tests/evaluation that do not validate the claims made about them;
- setup commands, dependency declarations, environment variables, and documentation that disagree with code;
- performance-sensitive work on synchronous or hot paths;
- missing failure handling, retry semantics, lifecycle cleanup, or state recovery supported by concrete call/data-flow evidence;
- observability that records metrics without making them actionable;
- architectural boundaries that make likely product changes unnecessarily expensive.

Confirm reachability and impact with call/data-flow traces rather than judging filenames in isolation.

### 5. Develop evidence-backed opportunities

For every proposed improvement, include:

- `Classification`: Confirmed gap or Inferred opportunity.
- `User value`: High, Medium, or Low, with a one-sentence reason.
- `Effort`: Small, Medium, or Large, based on affected components.
- `Dependencies`: Prerequisite recommendations, product decisions, migrations, or external services.
- `Evidence`: Exact files and qualified symbols/graph entities.
- `Recommendation`: Specific action tied to the evidence.
- `Expected result`: Observable product or engineering outcome.

Cover architectural, product, reliability, performance, observability, testing, documentation, and developer-experience dimensions only where the repository supplies evidence. It is acceptable for a dimension to have no recommendation.

### 6. Prioritize and sequence

Rank work using this order:

1. Correctness or broken core user flows.
2. High user value with small/medium effort.
3. Reliability and observability needed to operate existing capabilities.
4. Foundations required by multiple later improvements.
5. Larger product expansions and architectural changes.

Do not recommend parallel work that depends on an unfinished foundation. Identify prerequisites explicitly and group independent quick wins when useful.

## Evidence standard

Use evidence in this preferred order:

1. Runtime-reachable source symbols and graph traces.
2. Tests or evaluation code that exercises the behavior.
3. Configuration, schemas, and package manifests.
4. Documentation and product copy.
5. Inference from architecture or naming, clearly labeled as inference.

Format evidence precisely, for example:

`Evidence: src/api/session.py — SessionService.create_session; graph: CALLS -> TranscriptStore.append; docs/product.md — "session history".`

Never cite only a folder or make a claim such as “add tests,” “improve performance,” or “use caching” without naming the exact untested behavior, hot path, or repeated computation.

## Required output

Produce these sections in this order:

1. **Current system summary**
2. **What is already implemented well**
3. **Confirmed gaps**
4. **Improvement opportunities**
5. **Quick wins**
6. **High-impact improvements**
7. **Longer-term improvements**
8. **Recommended implementation sequence**
9. **Evidence index**

In the evidence index, map every recommendation identifier to its supporting files, modules, symbols, and graph entities. Keep confirmed findings separate from inferred opportunities throughout the report.

If the user requests a shorter review, preserve the evidence, classification, priority, effort, and dependency fields while reducing prose.
