"""
reports/patients.py - HTML reporting for patients data.
"""

from __future__ import annotations

import base64
import logging
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

def _embed_image(path: Path) -> str:
    """Read image file and return base64 src string."""
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    mime = "image/png"
    if path.suffix.lower() in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    return f"data:{mime};base64,{encoded}"

def _generate_datatable_js(df: pd.DataFrame, table_id: str = "analysisTable") -> str:
    """Generate HTML/JS for a DataTables interactive table."""
    # Convert DF to list of dicts for JSON
    data_json = df.to_json(orient="records")
    
    # Configure Columns explicitly for SearchPanes
    valid_panes = {'Sex', 'AgeGroup', 'Constraint', 'Analysis'}
    columns_config = []
    
    for col in df.columns:
        cfg = {"data": col, "title": col}
        if col in valid_panes:
             cfg["searchPanes"] = {"show": True}
        else:
             cfg["searchPanes"] = {"show": False}
        columns_config.append(cfg)
        
    columns_json = json.dumps(columns_config)

    html = f"""
    <div class="table-responsive">
        <div style="margin-bottom: 10px;">
            <label><strong>Filter by Minimum N (both groups > N):</strong> 
            <input type="number" id="minN" value="0" min="0" style="width: 80px; padding: 4px;"></label>
        </div>
        <table id="{table_id}" class="display table table-striped table-bordered" style="width:100%">
        </table>
    </div>
    <script>
        document.addEventListener('DOMContentLoaded', function () {{
            var data = {data_json};
            
            // Custom filtering function using rowData (4th arg) for safety
            $.fn.dataTable.ext.search.push(
                function(settings, data, dataIndex, rowData) {{
                    var min = parseInt($('#minN').val(), 10);
            
                    if (isNaN(min) || min <= 0) {{ return true; }}
                    
                    // Access properties directly from data object
                    var n1 = parseFloat(rowData['N1']) || 0; 
                    var n2 = parseFloat(rowData['N2']) || 0;
            
                    return n1 > min && n2 > min;
                }}
            );

            var table = $('#{table_id}').DataTable({{
                data: data,
                columns: {columns_json},
                pageLength: 25,
                dom: 'Pfrtip',
                buttons: ['copy', 'csv', 'excel'],
                searchPanes: {{
                    layout: 'columns-4',
                    initCollapsed: false,
                    cascadePanes: true,
                    viewTotal: true
                }},
                language: {{
                    searchPanes: {{
                        count: '{{total}} found',
                        countFiltered: '{{shown}} / {{total}}'
                    }}
                }}
            }});
            
            // Event listener to the two range filtering inputs to redraw on input
            $('#minN').on('keyup change', function () {{
                table.draw();
            }});
        }});
    </script>
    """
    return html

def generate_patients_report(
    df_clean: pd.DataFrame,
    validation_results: Dict[str, Any],
    figures_by_section: Dict[str, Dict[str, Path]],
    analysis_opportunities: pd.DataFrame,
    output_dir: Path,
    cleaning_stats: Dict[str, Any] = None
):
    """Generate standalone HTML report."""
    
    # Simple CSS
    css = """
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; color: #333; }
    .container { max_width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    h1, h2, h3 { color: #2c3e50; }
    h1 { border-bottom: 2px solid #3498db; padding-bottom: 10px; }
    .section { margin-bottom: 40px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }
    .card { border: 1px solid #ddd; padding: 15px; border-radius: 8px; background: white; }
    .card img { max-width: 100%; height: auto; }
    .stats-box { background: #ecf0f1; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
    table.clean-stats { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
    table.clean-stats th, table.clean-stats td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    table.clean-stats th { background-color: #f2f2f2; }
    .alert { padding: 15px; margin-bottom: 20px; border: 1px solid transparent; border-radius: 4px; }
    .alert-warning { color: #8a6d3b; background-color: #fcf8e3; border-color: #faebcc; }
    .alert-danger { color: #a94442; background-color: #f2dede; border-color: #ebccd1; }
    .alert-success { color: #3c763d; background-color: #dff0d8; border-color: #d6e9c6; }
    """

    # CDN Links for DataTables/JQuery + SearchPanes + Select
    head = f"""
    <head>
        <title>Patients Data Report</title>
        <meta charset="UTF-8">
        <style>{css}</style>
        <!-- DataTables CSS -->
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.11.5/css/jquery.dataTables.css">
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/buttons/2.2.2/css/buttons.dataTables.min.css">
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/searchpanes/2.0.0/css/searchPanes.dataTables.min.css">
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/select/1.3.4/css/select.dataTables.min.css">
        
        <!-- jQuery & DataTables JS -->
        <script type="text/javascript" charset="utf8" src="https://code.jquery.com/jquery-3.5.1.js"></script>
        <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.js"></script>
        <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/buttons/2.2.2/js/dataTables.buttons.min.js"></script>
        <script type="text/javascript" charset="utf8" src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.1.3/jszip.min.js"></script>
        <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/buttons/2.2.2/js/buttons.html5.min.js"></script>
        <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/searchpanes/2.0.0/js/dataTables.searchPanes.min.js"></script>
        <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/select/1.3.4/js/dataTables.select.min.js"></script>
    </head>
    """

    # 1. Cleaning Stats HTML
    cleaning_html = ""
    if cleaning_stats:
        cleaning_html = "<div class='stats-box'><h3>Data Cleaning Log</h3><table class='clean-stats'>"
        cleaning_html += "<tr><th>Step</th><th>Filtered/Dropped</th><th>Remaining Subjects</th></tr>"
        
        # Row 1: Initial
        n_init = cleaning_stats.get('n_initial', '?')
        cleaning_html += f"<tr><td>Initial CSV Load</td><td>-</td><td><strong>{n_init}</strong></td></tr>"
        
        # '0 (potentiel)'
        n_pot_dropped = cleaning_stats.get('n_potential_dropped', 0)
        n_after_pot = cleaning_stats.get('n_after_potential', '?')
        cleaning_html += f"<tr><td>Drop '0 (potentiel)'</td><td>{n_pot_dropped} dropped</td><td>{n_after_pot}</td></tr>"
        
        # Mismatches
        n_mis_dropped = cleaning_stats.get('n_mismatches_dropped', 0)
        n_after_mis = cleaning_stats.get('n_after_mismatch', '?')
        cleaning_html += f"<tr><td>Medication Mismatches</td><td>{n_mis_dropped} dropped</td><td>{n_after_mis}</td></tr>"
        
        # Duplicates
        n_dup_dropped = cleaning_stats.get('n_duplicates_dropped', 0)
        n_fin = cleaning_stats.get('n_final', '?')
        cleaning_html += f"<tr><td>Duplicate Pt IDs</td><td>{n_dup_dropped} dropped</td><td><strong>{n_fin}</strong></td></tr>"
        
        cleaning_html += "</table></div>"

    # BIDS Coverage Section (Modified to not repeat total subjects if stats present)
    bids_html = "<div class='stats-box'><h3>Dataset Coverage</h3>"
    if not cleaning_stats:
        bids_html += f"<p><strong>Total Subjects in CSV:</strong> {len(df_clean)}</p>"
    
    bids_html += f"<p><strong>Subjects with BIDS Folders:</strong> {len(validation_results.get('bids_present', []))}</p>"
    
    missing_count = validation_results.get('missing_count', 0)
    if missing_count > 0:
        bids_html += f"<div class='alert alert-warning'><strong>Warning:</strong> {missing_count} subjects missing from BIDS folder.</div>"
        missing_ids = validation_results.get("missing_study_ids", [])
        if missing_ids:
             # Limit output
             display_ids = missing_ids[:20]
             suffix = "..." if len(missing_ids) > 20 else ""
             bids_html += f"<p><small>Missing IDs: {display_ids}{suffix}</small></p>"
    else:
        bids_html += "<div class='alert alert-success'>All subjects present in BIDS.</div>"
    
    bids_html += "</div>"
    
    # Figures Section (Grouped by Section)
    figs_html = ""
    for section_name, figs_dict in figures_by_section.items():
        if not figs_dict: continue
        
        figs_html += f"<div class='section'><h2>{section_name} Visualizations</h2><div class='grid'>"
        
        for key, path in figs_dict.items():
            if path.exists():
                src = _embed_image(path)
                # Pretty Title: "diagnosis_prevalence" -> "Diagnosis Prevalence"
                title = key.replace("_", " ").title()
                figs_html += f"<div class='card'><h3>{title}</h3><img src='{src}'></div>"
                
        figs_html += "</div></div>"

    # Analysis Table Section
    table_html = "<div class='section'><h2>Analysis Opportunities</h2>"
    table_html += "<p>Search and filter potential analysis groups (N > 0).</p>"
    if not analysis_opportunities.empty:
        table_html += _generate_datatable_js(analysis_opportunities)
    else:
        table_html += "<p>No analysis opportunities found.</p>"
    table_html += "</div>"

    body = f"""
    <body>
        <div class="container">
            <h1>Patients Data Report</h1>
            <p>Generated logic from <code>qc/patients.py</code></p>
            {cleaning_html}
            {bids_html}
            {figs_html}
            {table_html}
        </div>
    </body>
    """

    report_path = output_dir / "patients_report.html"
    report_path.write_text(f"<!DOCTYPE html><html>{head}{body}</html>", encoding="utf-8")
    logging.info(f"Saved HTML report: {report_path}")

