"""Cohort report generation over cleaned patient metadata."""

from __future__ import annotations

import html
import json
import uuid
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
from coco_pipe.report.core import Element, ImageElement, Report, Section, TableElement


def _add_images(section: Section, figures: Sequence[tuple[str, Path]]) -> None:
    for title, path in figures:
        if path.exists():
            section.add_element(ImageElement(str(path), caption=title))


def _add_optional_table(section: Section, data: pd.DataFrame, title: str) -> None:
    if not data.empty:
        section.add_element(TableElement(data, title=title))


class InteractiveOpportunitiesElement(Element):
    """Interactive valid-opportunities table with client-side filters and paging."""

    def __init__(self, data: pd.DataFrame, title: str = "Valid Analysis Opportunities") -> None:
        self.data = data
        self.title = title
        self.element_id = f"opps-{uuid.uuid4().hex[:8]}"

    def render(self) -> str:
        no_filter_value = "__ALL__"
        df = self.data.copy()
        if df.empty:
            return (
                '<div class="my-4">'
                f'<h4 class="text-sm font-semibold text-gray-700 dark:text-gray-300 uppercase tracking-wide">{html.escape(self.title)}</h4>'
                '<p class="text-sm text-gray-600 dark:text-gray-400 mt-2">No valid analysis opportunities.</p>'
                "</div>"
            )

        records = json.dumps(df.astype(object).where(pd.notna(df), None).to_dict(orient="records"))
        columns = json.dumps(df.columns.tolist())

        select_columns = ["Sex", "AgeGroup", "Constraint", "Analysis"]
        options = {
            column: sorted(df[column].dropna().astype(str).unique().tolist())
            for column in select_columns
            if column in df.columns
        }
        options_json = json.dumps(options)
        title = html.escape(self.title)
        element_id = self.element_id

        return f"""
<div id="{element_id}" class="my-4">
  <div class="flex justify-between items-center mb-3">
    <h4 class="text-sm font-semibold text-gray-700 dark:text-gray-300 uppercase tracking-wide">{title}</h4>
  </div>
  <div class="grid grid-cols-1 md:grid-cols-4 gap-3 mb-3 text-sm">
    <label class="block">
      <span class="block text-gray-600 dark:text-gray-400 mb-1">Sex</span>
      <select data-filter="Sex" class="w-full border rounded px-2 py-1 dark:bg-gray-900 dark:border-gray-700"></select>
    </label>
    <label class="block">
      <span class="block text-gray-600 dark:text-gray-400 mb-1">Age Group</span>
      <select data-filter="AgeGroup" class="w-full border rounded px-2 py-1 dark:bg-gray-900 dark:border-gray-700"></select>
    </label>
    <label class="block">
      <span class="block text-gray-600 dark:text-gray-400 mb-1">Constraint</span>
      <select data-filter="Constraint" class="w-full border rounded px-2 py-1 dark:bg-gray-900 dark:border-gray-700"></select>
    </label>
    <label class="block">
      <span class="block text-gray-600 dark:text-gray-400 mb-1">Analysis</span>
      <select data-filter="Analysis" class="w-full border rounded px-2 py-1 dark:bg-gray-900 dark:border-gray-700"></select>
    </label>
  </div>
  <div class="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4 text-sm">
    <label class="block">
      <span class="block text-gray-600 dark:text-gray-400 mb-1">Rows per Page</span>
      <select data-page-size class="w-full border rounded px-2 py-1 dark:bg-gray-900 dark:border-gray-700">
        <option value="25">25</option>
        <option value="50">50</option>
        <option value="100">100</option>
      </select>
    </label>
    <label class="block">
      <span class="block text-gray-600 dark:text-gray-400 mb-1">Min per Group</span>
      <input
        type="number"
        min="0"
        step="1"
        value="0"
        data-min-group
        class="w-full border rounded px-2 py-1 dark:bg-gray-900 dark:border-gray-700"
      />
    </label>
    <label class="block">
      <span class="block text-gray-600 dark:text-gray-400 mb-1">Order By</span>
      <select data-sort-by class="w-full border rounded px-2 py-1 dark:bg-gray-900 dark:border-gray-700">
        <option value="Analysis">Analysis</option>
        <option value="Constraint">Constraint</option>
        <option value="cohort_n">Cohort N</option>
        <option value="N1">N1</option>
        <option value="N2">N2</option>
      </select>
    </label>
    <label class="block">
      <span class="block text-gray-600 dark:text-gray-400 mb-1">Direction</span>
      <select data-sort-direction class="w-full border rounded px-2 py-1 dark:bg-gray-900 dark:border-gray-700">
        <option value="desc">Descending</option>
        <option value="asc">Ascending</option>
      </select>
    </label>
  </div>
  <div class="flex justify-between items-center mb-2 text-sm text-gray-600 dark:text-gray-400">
    <div data-status></div>
    <div class="space-x-2">
      <button data-prev class="px-3 py-1 border rounded disabled:opacity-40">Previous</button>
      <button data-next class="px-3 py-1 border rounded disabled:opacity-40">Next</button>
    </div>
  </div>
  <div class="overflow-x-auto">
    <table class="min-w-full divide-y divide-gray-200 dark:divide-gray-700 border dark:border-gray-700 text-sm">
      <thead class="bg-gray-50 dark:bg-gray-800">
        <tr data-header-row></tr>
      </thead>
      <tbody data-body class="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700"></tbody>
    </table>
  </div>
</div>
<script>
(function() {{
  const root = document.getElementById({json.dumps(element_id)});
  if (!root) return;
  const rows = {records};
  const columns = {columns};
  const filterOptions = {options_json};
  const filters = {{}};
  let page = 0;
  let pageSize = 25;

  const headerRow = root.querySelector('[data-header-row]');
  const body = root.querySelector('[data-body]');
  const status = root.querySelector('[data-status]');
  const prevBtn = root.querySelector('[data-prev]');
  const nextBtn = root.querySelector('[data-next]');
  const pageSizeSelect = root.querySelector('[data-page-size]');
  const minGroupInput = root.querySelector('[data-min-group]');
  const sortBySelect = root.querySelector('[data-sort-by]');
  const sortDirectionSelect = root.querySelector('[data-sort-direction]');

  headerRow.innerHTML = columns.map(
    (col) => `<th class="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">${{col}}</th>`
  ).join('');

  Object.entries(filterOptions).forEach(([column, values]) => {{
    const select = root.querySelector(`select[data-filter="${{column}}"]`);
    if (!select) return;
    select.innerHTML = [`<option value="${no_filter_value}">Select</option>`]
      .concat(values.map((value) => `<option value="${{String(value)}}">${{String(value)}}</option>`))
      .join('');
    filters[column] = {json.dumps(no_filter_value)};
    select.addEventListener('change', () => {{
      filters[column] = select.value;
      page = 0;
      render();
    }});
  }});

  pageSizeSelect.addEventListener('change', () => {{
    pageSize = parseInt(pageSizeSelect.value, 10);
    page = 0;
    render();
  }});

  minGroupInput.addEventListener('input', () => {{
    page = 0;
    render();
  }});

  sortBySelect.addEventListener('change', () => {{
    page = 0;
    render();
  }});

  sortDirectionSelect.addEventListener('change', () => {{
    page = 0;
    render();
  }});

  prevBtn.addEventListener('click', () => {{
    if (page > 0) {{
      page -= 1;
      render();
    }}
  }});

  nextBtn.addEventListener('click', () => {{
    const totalPages = Math.max(1, Math.ceil(filteredRows().length / pageSize));
    if (page < totalPages - 1) {{
      page += 1;
      render();
    }}
  }});

  function filteredRows() {{
    const minGroup = Math.max(0, parseInt(minGroupInput.value || '0', 10) || 0);
    const filtered = rows.filter((row) => Object.entries(filters).every(([column, value]) => {{
      if (value === {json.dumps(no_filter_value)}) return true;
      return String(row[column]) === value;
    }}) && Number(row.N1 ?? 0) >= minGroup && Number(row.N2 ?? 0) >= minGroup);

    const sortBy = sortBySelect.value;
    const direction = sortDirectionSelect.value === 'asc' ? 1 : -1;
    const numericColumns = new Set(['cohort_n', 'N1', 'N2']);

    return filtered.slice().sort((left, right) => {{
      const leftValue = left[sortBy];
      const rightValue = right[sortBy];
      if (numericColumns.has(sortBy)) {{
        return direction * ((Number(leftValue ?? 0)) - (Number(rightValue ?? 0)));
      }}
      return direction * String(leftValue ?? '').localeCompare(String(rightValue ?? ''));
    }});
  }}

  function render() {{
    const filtered = filteredRows();
    const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
    if (page >= totalPages) page = totalPages - 1;
    const start = page * pageSize;
    const paged = filtered.slice(start, start + pageSize);

    body.innerHTML = paged.map((row) => {{
      const tds = columns.map((column) => {{
        const value = row[column] ?? '';
        return `<td class="px-4 py-3 whitespace-nowrap text-gray-700 dark:text-gray-300">${{value}}</td>`;
      }}).join('');
      return `<tr>${{tds}}</tr>`;
    }}).join('');

    status.textContent = `Showing ${{filtered.length === 0 ? 0 : start + 1}}-${{Math.min(start + pageSize, filtered.length)}} of ${{filtered.length}} rows`;
    prevBtn.disabled = page === 0;
    nextBtn.disabled = page >= totalPages - 1 || filtered.length === 0;
  }}

  render();
}})();
</script>
"""


class InteractiveRecruitmentElement(Element):
    """Interactive milestone explorer for recruitment projections."""

    def __init__(self, data: pd.DataFrame, title: str = "Recruitment Milestone Explorer") -> None:
        self.data = data
        self.title = title
        self.element_id = f"recruit-{uuid.uuid4().hex[:8]}"

    def render(self) -> str:
        no_filter_value = "__ALL__"
        df = self.data.copy()
        if df.empty:
            return (
                '<div class="my-4">'
                f'<h4 class="text-sm font-semibold text-gray-700 dark:text-gray-300 uppercase tracking-wide">{html.escape(self.title)}</h4>'
                '<p class="text-sm text-gray-600 dark:text-gray-400 mt-2">No recruitment projections available.</p>'
                "</div>"
            )

        display_columns = [
            "milestone",
            "family",
            "analysis",
            "constraint",
            "group_1",
            "group_2",
            "projected_n1",
            "projected_n2",
            "required_n1",
            "required_n2",
            "shortfall_n1",
            "shortfall_n2",
            "limiting_group",
        ]
        df = df[display_columns].copy()
        records = json.dumps(df.astype(object).where(pd.notna(df), None).to_dict(orient="records"))
        columns = json.dumps(df.columns.tolist())
        title = html.escape(self.title)
        element_id = self.element_id

        milestone_options = sorted(df["milestone"].dropna().astype(int).unique().tolist())
        family_options = sorted(df["family"].dropna().astype(str).unique().tolist())
        constraint_options = sorted(df["constraint"].dropna().astype(str).unique().tolist())
        analysis_options = sorted(df["analysis"].dropna().astype(str).unique().tolist())
        return f"""
<div id="{element_id}" class="my-4">
  <div class="flex justify-between items-center mb-3">
    <h4 class="text-sm font-semibold text-gray-700 dark:text-gray-300 uppercase tracking-wide">{title}</h4>
  </div>
  <div class="grid grid-cols-1 md:grid-cols-4 gap-3 mb-3 text-sm">
    <label class="block">
      <span class="block text-gray-600 dark:text-gray-400 mb-1">Milestone</span>
      <select data-filter="milestone" class="w-full border rounded px-2 py-1 dark:bg-gray-900 dark:border-gray-700">
        {''.join(f'<option value="{value}">{value}</option>' for value in milestone_options)}
      </select>
    </label>
    <label class="block">
      <span class="block text-gray-600 dark:text-gray-400 mb-1">Family</span>
      <select data-filter="family" class="w-full border rounded px-2 py-1 dark:bg-gray-900 dark:border-gray-700">
        <option value="{no_filter_value}">Select</option>
        {''.join(f'<option value="{html.escape(value)}">{html.escape(value)}</option>' for value in family_options)}
      </select>
    </label>
    <label class="block">
      <span class="block text-gray-600 dark:text-gray-400 mb-1">Constraint</span>
      <select data-filter="constraint" class="w-full border rounded px-2 py-1 dark:bg-gray-900 dark:border-gray-700">
        <option value="{no_filter_value}">Select</option>
        {''.join(f'<option value="{html.escape(value)}">{html.escape(value)}</option>' for value in constraint_options)}
      </select>
    </label>
    <label class="block">
      <span class="block text-gray-600 dark:text-gray-400 mb-1">Analysis</span>
      <select data-filter="analysis" class="w-full border rounded px-2 py-1 dark:bg-gray-900 dark:border-gray-700">
        <option value="{no_filter_value}">Select</option>
        {''.join(f'<option value="{html.escape(value)}">{html.escape(value)}</option>' for value in analysis_options)}
      </select>
    </label>
  </div>
  <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4 text-sm">
    <label class="block">
      <span class="block text-gray-600 dark:text-gray-400 mb-1">Rows per Page</span>
      <select data-page-size class="w-full border rounded px-2 py-1 dark:bg-gray-900 dark:border-gray-700">
        <option value="25">25</option>
        <option value="50">50</option>
        <option value="100">100</option>
      </select>
    </label>
    <label class="block">
      <span class="block text-gray-600 dark:text-gray-400 mb-1">Order By</span>
      <select data-sort-by class="w-full border rounded px-2 py-1 dark:bg-gray-900 dark:border-gray-700">
        <option value="shortfall_n1">Shortfall N1</option>
        <option value="shortfall_n2">Shortfall N2</option>
        <option value="projected_n1">Projected N1</option>
        <option value="projected_n2">Projected N2</option>
        <option value="analysis">Analysis</option>
      </select>
    </label>
  </div>
  <div class="flex justify-between items-center mb-2 text-sm text-gray-600 dark:text-gray-400">
    <div data-status></div>
    <div class="space-x-2">
      <button data-prev class="px-3 py-1 border rounded disabled:opacity-40">Previous</button>
      <button data-next class="px-3 py-1 border rounded disabled:opacity-40">Next</button>
    </div>
  </div>
  <div class="overflow-x-auto">
    <table class="min-w-full divide-y divide-gray-200 dark:divide-gray-700 border dark:border-gray-700 text-sm">
      <thead class="bg-gray-50 dark:bg-gray-800">
        <tr data-header-row></tr>
      </thead>
      <tbody data-body class="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700"></tbody>
    </table>
  </div>
</div>
<script>
(function() {{
  const root = document.getElementById({json.dumps(element_id)});
  if (!root) return;
  const rows = {records};
  const columns = {columns};
  const noFilterValue = {json.dumps(no_filter_value)};
  let page = 0;
  let pageSize = 25;

  const filters = {{
    milestone: String({milestone_options[0] if milestone_options else '""'}),
    family: noFilterValue,
    constraint: noFilterValue,
    analysis: noFilterValue,
  }};

  const headerRow = root.querySelector('[data-header-row]');
  const body = root.querySelector('[data-body]');
  const status = root.querySelector('[data-status]');
  const prevBtn = root.querySelector('[data-prev]');
  const nextBtn = root.querySelector('[data-next]');
  const pageSizeSelect = root.querySelector('[data-page-size]');
  const sortBySelect = root.querySelector('[data-sort-by]');

  headerRow.innerHTML = columns.map(
    (col) => `<th class="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">${{col}}</th>`
  ).join('');

  root.querySelectorAll('select[data-filter]').forEach((select) => {{
    const key = select.getAttribute('data-filter');
    select.value = filters[key];
    select.addEventListener('change', () => {{
      filters[key] = select.value;
      page = 0;
      render();
    }});
  }});

  pageSizeSelect.addEventListener('change', () => {{
    pageSize = parseInt(pageSizeSelect.value, 10);
    page = 0;
    render();
  }});

  sortBySelect.addEventListener('change', () => {{
    page = 0;
    render();
  }});

  prevBtn.addEventListener('click', () => {{
    if (page > 0) {{
      page -= 1;
      render();
    }}
  }});

  nextBtn.addEventListener('click', () => {{
    const totalPages = Math.max(1, Math.ceil(filteredRows().length / pageSize));
    if (page < totalPages - 1) {{
      page += 1;
      render();
    }}
  }});

  function filteredRows() {{
    const filtered = rows.filter((row) => {{
      if (String(row.milestone) !== filters.milestone) return false;
      if (filters.family !== noFilterValue && String(row.family) !== filters.family) return false;
      if (filters.constraint !== noFilterValue && String(row.constraint) !== filters.constraint) return false;
      if (filters.analysis !== noFilterValue && String(row.analysis) !== filters.analysis) return false;
      return true;
    }});

    const sortBy = sortBySelect.value;
    const numericColumns = new Set(['milestone', 'projected_n1', 'projected_n2', 'required_n1', 'required_n2', 'shortfall_n1', 'shortfall_n2']);
    return filtered.slice().sort((left, right) => {{
      if (numericColumns.has(sortBy)) {{
        return Number(right[sortBy] ?? 0) - Number(left[sortBy] ?? 0);
      }}
      return String(left[sortBy] ?? '').localeCompare(String(right[sortBy] ?? ''));
    }});
  }}

  function render() {{
    const filtered = filteredRows();
    const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
    if (page >= totalPages) page = totalPages - 1;
    const start = page * pageSize;
    const paged = filtered.slice(start, start + pageSize);

    body.innerHTML = paged.map((row) => {{
      const tds = columns.map((column) => `<td class="px-4 py-3 whitespace-nowrap text-gray-700 dark:text-gray-300">${{row[column] ?? ''}}</td>`).join('');
      return `<tr>${{tds}}</tr>`;
    }}).join('');

    status.textContent = `Showing ${{filtered.length === 0 ? 0 : start + 1}}-${{Math.min(start + pageSize, filtered.length)}} of ${{filtered.length}} rows for milestone ${{filters.milestone}}`;
    prevBtn.disabled = page === 0;
    nextBtn.disabled = page >= totalPages - 1 || filtered.length === 0;
  }}

  render();
}})();
</script>
"""


class InteractiveRecruitmentPoolsElement(Element):
    """Interactive recruitment-pool rollup table with milestone selector."""

    def __init__(self, data: pd.DataFrame, title: str = "Recruitment Pools") -> None:
        self.data = data
        self.title = title
        self.element_id = f"pool-{uuid.uuid4().hex[:8]}"

    def render(self) -> str:
        df = self.data.copy()
        if df.empty:
            return ""

        display_columns = [
            "pool",
            "current_n",
            "target_n",
            "raw_gap",
            "child_planned",
            "net_recruit_needed",
        ]
        records = json.dumps(df[["milestone", *display_columns]].astype(object).where(pd.notna(df), None).to_dict(orient="records"))
        columns = json.dumps(display_columns)
        milestone_options = sorted(df["milestone"].dropna().astype(int).unique().tolist())
        element_id = self.element_id
        title = html.escape(self.title)

        return f"""
<div id="{element_id}" class="my-4">
  <div class="flex justify-between items-center mb-3">
    <h4 class="text-sm font-semibold text-gray-700 dark:text-gray-300 uppercase tracking-wide">{title}</h4>
  </div>
  <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4 text-sm">
    <label class="block">
      <span class="block text-gray-600 dark:text-gray-400 mb-1">Milestone</span>
      <select data-milestone class="w-full border rounded px-2 py-1 dark:bg-gray-900 dark:border-gray-700">
        {''.join(f'<option value="{value}">{value}</option>' for value in milestone_options)}
      </select>
    </label>
  </div>
  <div class="overflow-x-auto">
    <table class="min-w-full divide-y divide-gray-200 dark:divide-gray-700 border dark:border-gray-700 text-sm">
      <thead class="bg-gray-50 dark:bg-gray-800">
        <tr data-header-row></tr>
      </thead>
      <tbody data-body class="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700"></tbody>
    </table>
  </div>
</div>
<script>
(function() {{
  const root = document.getElementById({json.dumps(element_id)});
  if (!root) return;
  const rows = {records};
  const columns = {columns};
  let milestone = String({milestone_options[0] if milestone_options else '""'});

  const headerRow = root.querySelector('[data-header-row]');
  const body = root.querySelector('[data-body]');
  const milestoneSelect = root.querySelector('[data-milestone]');

  headerRow.innerHTML = columns.map(
    (col) => `<th class="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">${{col}}</th>`
  ).join('');

  milestoneSelect.addEventListener('change', () => {{
    milestone = milestoneSelect.value;
    render();
  }});

  function render() {{
    const filtered = rows.filter((row) => String(row.milestone) === milestone);

    body.innerHTML = filtered.map((row) => {{
      const tds = columns.map((column) => `<td class="px-4 py-3 whitespace-nowrap text-gray-700 dark:text-gray-300">${{row[column] ?? ''}}</td>`).join('');
      return `<tr>${{tds}}</tr>`;
    }}).join('');
  }}

  render();
}})();
</script>
"""


def generate_cohort_report(
    output_path: Path,
    report_title: str,
    cohort_name: str,
    cohort_markdown: str,
    cohort_summary_df: pd.DataFrame,
    provenance_reason_df: pd.DataFrame,
    provenance_source_df: pd.DataFrame,
    diagnosis_df: pd.DataFrame,
    combined_diagnosis_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
    medication_df: pd.DataFrame,
    valid_opportunities_df: pd.DataFrame,
    figures_by_section: Mapping[str, Sequence[tuple[str, Path]]],
    recruitment_markdown: str | None = None,
    recruitment_projection_df: pd.DataFrame | None = None,
    recruitment_summary_df: pd.DataFrame | None = None,
    recruitment_pools_df: pd.DataFrame | None = None,
) -> Path:
    report = Report(title=report_title)

    cohort_definition = Section("Cohort Definition", icon="🎯")
    cohort_definition.add_markdown(
        f"Phase 1 cohort report for **{cohort_name}**, built directly from "
        "`patients_metadata_clean.csv`."
    )
    cohort_definition.add_markdown(cohort_markdown)
    _add_optional_table(cohort_definition, cohort_summary_df, "Cohort Summary")
    _add_images(cohort_definition, figures_by_section.get("Cohort Definition", []))
    report.add_section(cohort_definition)

    provenance = Section("Provenance", icon="🧹")
    provenance.add_markdown(
        "These summaries come from `patients_metadata_removed.json` and describe the "
        "metadata-build drops that happened before this report."
    )
    _add_optional_table(provenance, provenance_reason_df, "Removed Rows by Reason")
    _add_optional_table(provenance, provenance_source_df, "Removed Rows by Source Dataset")
    report.add_section(provenance)

    diagnosis = Section("Diagnosis and Demographics", icon="🧾")
    _add_optional_table(diagnosis, diagnosis_df, "Diagnosis Summary")
    for title, path in figures_by_section.get("Diagnosis and Demographics", []):
        if title == "Diagnosis Prevalence" and path.exists():
            diagnosis.add_element(ImageElement(str(path), caption=title))
    _add_optional_table(diagnosis, combined_diagnosis_df, "Combined Diagnosis Summary")
    for title, path in figures_by_section.get("Diagnosis and Demographics", []):
        if title == "Combined Diagnosis Counts" and path.exists():
            diagnosis.add_element(ImageElement(str(path), caption=title))
    _add_optional_table(diagnosis, demographics_df, "Demographics Summary")
    for title, path in figures_by_section.get("Diagnosis and Demographics", []):
        if title in {"Sex by Age Group", "Age by Combined Diagnosis"} and path.exists():
            diagnosis.add_element(ImageElement(str(path), caption=title))
    report.add_section(diagnosis)

    medication = Section("Medication and Drug Resistance", icon="💊")
    _add_optional_table(medication, medication_df, "Medication Summary")
    _add_images(medication, figures_by_section.get("Medication and Drug Resistance", []))
    report.add_section(medication)

    opportunities = Section("Analysis Opportunities", icon="🧠")
    opportunities.add_element(InteractiveOpportunitiesElement(valid_opportunities_df))
    report.add_section(opportunities)

    if recruitment_summary_df is not None:
        recruitment = Section("Recruitment Strategy", icon="📈")
        if recruitment_markdown:
            recruitment.add_markdown(recruitment_markdown)
        if recruitment_projection_df is not None and not recruitment_projection_df.empty:
            recruitment.add_element(InteractiveRecruitmentElement(recruitment_projection_df))
        if recruitment_pools_df is not None and not recruitment_pools_df.empty:
            recruitment.add_element(InteractiveRecruitmentPoolsElement(recruitment_pools_df))
        report.add_section(recruitment)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(str(output_path))
    return output_path
