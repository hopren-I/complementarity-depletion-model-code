"""Source summary rows for the current InSb prediction table (Table V)."""
import pandas as pd
from insb_io import ROOT
P=ROOT.parent

def write_table8(write_csv=True):
    """Display capacitance errors; retain inclusion diagnostics in the full CSV."""
    summary=pd.read_csv(ROOT/'insb_summary.csv')
    # Generated table uses black text, including tabular headers and numbers.
    mainrows=summary[(summary.window=='low_04_08')].set_index('condition')
    lowrows=pd.read_csv(ROOT/'insb_paired_low_loss_summary.csv').set_index('condition')
    tablelabels={'en':['2 MHz, interleaved holdout','2 MHz, contiguous holdout','100 kHz, forward','5 kHz, forward','2 MHz, reverse','2 MHz, forward','2 MHz, reverse'],
                 'zh':['2 MHz前扫，交错留出','2 MHz前扫，连续留出','100 kHz前扫','5 kHz前扫','2 MHz反扫','2 MHz前扫','2 MHz反扫']}
    order=['same_2MHz','blocked_same_2MHz','forward_100000Hz','forward_5000Hz','reverse_2e+06Hz']
    table_records=[]
    for lang in ['en','zh']:
        header=['Condition','Devices',r'\shortstack{C error (\%)\\median [IQR]}'] if lang=='en' else ['测量条件','器件数',r'\shortstack{电容误差/\%\\中位数[四分位区间]}']
        lines=[r'\begingroup\color{black}\small\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{lrr}',r'\toprule',' & '.join(header)+r' \\',r'\midrule']
        for k,cond in enumerate(order):
            r=mainrows.loc[cond];a=100*r.mape_median;l=100*r.mape_q25;u=100*r.mape_q75
            lines.append(f'{tablelabels[lang][k]} & {int(r.n_devices)} & {a:.2f} [{l:.2f}, {u:.2f}]'+r' \\')
            if lang=='en':table_records.append(dict(cohort='all eligible',condition=cond,**r.to_dict()))
        title='Same paired low-loss cohort' if lang=='en' else '同一双低损耗配对队列'
        lines.extend([r'\midrule',r'\multicolumn{3}{l}{'+title+r'} \\'])
        for k,cond in enumerate(['same_2MHz','reverse_2e+06Hz']):
            r=lowrows.loc[cond];a=100*r.mape_median;l=100*r.mape_q25;u=100*r.mape_q75
            lines.append(f'{tablelabels[lang][k+5]} & {int(r.n_devices)} & {a:.2f} [{l:.2f}, {u:.2f}]'+r' \\')
            if lang=='en':table_records.append(dict(cohort='paired low loss',condition=cond,**r.to_dict()))
        counts=[int(mainrows.loc[c].n_test_points) for c in order]
        paired_counts=[int(lowrows.loc[c].n_test_points) for c in ['same_2MHz','reverse_2e+06Hz']]
        note1=('Test points (first five rows): ' if lang=='en' else '前五行测试点数：')+', '.join(map(str,counts))+'.'
        note2=('Paired rows: ' if lang=='en' else '配对两行测试点数：')+', '.join(map(str,paired_counts))+'.'
        note3='All rows: zero undefined C predictions; no clipping.' if lang=='en' else '各行未定义电容预测点数均为0；未作截断。'
        lines.extend([r'\midrule']+[r'\multicolumn{3}{l}{'+note+r'} \\' for note in [note1,note2,note3]]+[r'\bottomrule',r'\end{tabular}',r'\endgroup'])
        (P/f'tables/table8_{lang}.tex').write_text('\n'.join(lines)+'\n')
    if write_csv:
        pd.DataFrame(table_records).to_csv(ROOT/'insb_table8_values.csv',index=False)

