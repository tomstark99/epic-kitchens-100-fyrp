import argparse
import os

from pathlib import Path
import pandas as pd

import dash
from dash import Dash
from dash.exceptions import PreventUpdate
import flask

from apps.esv_dashboard.visualisation import Visualiser
from apps.esv_dashboard.visualisation_mf import VisualiserMF
from apps.esv_dashboard.result import Result, ShapleyValueResultsMTRN, ShapleyValueResultsMF

parser = argparse.ArgumentParser(
    description="Run web-based ESV visualisation tool",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("esvs_root", type=Path, help="Path to folder containing extracted ESVs")
parser.add_argument("dataset_root", type=Path, help="Path dataset folder of videos")
parser.add_argument("labels_root", type=Path, help="Path to labels root")
parser.add_argument("--debug", default=True, type=bool, help="Enable Dash debug capabilities")
parser.add_argument("--port", default=8050, type=int, help="Port for webserver to listen on")
parser.add_argument("--host", default="localhost", help="Host to bind to")
parser.add_argument("--motion-former", default=False, action='store_true', help="Display motion former ESVs")
parser.add_argument("--test", default=False, action='store_true', help="Display ESVs for test set")

def main(args):

    args = parser.parse_args()
    dataset_dir: Path = args.dataset_root

    colours = {
        'rgb': {
            'yellow_20': 'rgba(244,160,0,0.1)',
            'blue_20': 'rgba(66,133,244,0.05)'
        },
        'hex': {
            'red': '#DB4437',
            'red_mf': '#ff0000',
            'blue': '#4285F4',
            'blue_mf': '#000cff',
            'yellow': '#F4B400',
            'green': '#0F9D58'
        }
    }
    verbs = pd.read_csv(args.labels_root / 'EPIC_100_verb_classes.csv')
    nouns = pd.read_csv(args.labels_root / 'EPIC_100_noun_classes.csv')

    verb2str = pd.Series(verbs.key.values,index=verbs.id).to_dict()
    noun2str = pd.Series(nouns.key.values,index=nouns.id).to_dict()

    verb_noun = pd.read_pickle(args.labels_root / 'verb_noun.pkl')
    verb_noun_classes = pd.read_pickle(args.labels_root / 'verb_noun_classes.pkl')
    verb_noun_narration = pd.read_pickle(args.labels_root /'verb_noun_classes_narration.pkl')

    if args.test:
        results_dict_mtrn = pd.read_pickle(args.esvs_root / 'f_val_mtrn-esv-min_frames=1-max_frames=8.pkl')
        results_dict_mf = pd.read_pickle(args.esvs_root / 'f_val_mf-esv-min_frames=1-max_frames=8.pkl')
        features_path = Path('datasets/epic-100/features/9668_val_features.pkl')
    else:
        results_dict_mtrn = pd.read_pickle(args.esvs_root / 'f_train_mtrn-esv-min_frames=1-max_frames=8.pkl')
        results_dict_mf = pd.read_pickle(args.esvs_root / 'f_train_mf-esv-min_frames=1-max_frames=8.pkl')
        features_path = Path('datasets/epic-100/features/67217_train_features.pkl')

    if args.motion_former:
        labels_dict = pd.read_pickle(features_path)['labels']

    title = "ESV Dashboard"
    
    results = ShapleyValueResultsMTRN(results_dict_mtrn)

    if args.motion_former:
        results_mf = ShapleyValueResultsMF(results_dict_mf, labels_dict)
        visualisation = VisualiserMF(
            results,
            results_mf,
            colours, 
            verb2str, 
            noun2str, 
            verb_noun,
            verb_noun_classes,
            verb_noun_narration, 
            dataset_dir, 
            title=title
        )
    else:
        visualisation = Visualiser(
            results,
            colours, 
            verb2str, 
            noun2str, 
            verb_noun,
            verb_noun_classes,
            verb_noun_narration, 
            dataset_dir, 
            title=title
        )

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = Dash(
        __name__,
        title="ESV Visualiser",
        update_title="Updating..." if args.debug else None,
        external_stylesheets=external_stylesheets,
    )

    visualisation.attach_to_app(app)
    app.run_server(host=args.host, debug=args.debug, port=args.port)

if __name__ == "__main__":
    main(parser.parse_args())

