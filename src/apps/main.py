"""routing command-line stage choices to the right module"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["1", "2", "2b", "3", "3b", "4", "5", "6",
                                 "all"])
    parser.add_argument("--chat-mode", default="cli",
                        choices=["cli", "gradio", "both"])
    args = parser.parse_args()

    # importing lazily so each stage only loads what it needs
    if args.stage in ("1", "all"):
        from pipeline.data_loading import stage_1_load
        stage_1_load()
    if args.stage in ("2", "all"):
        from pipeline.features import stage_2_features
        stage_2_features()
    if args.stage in ("2b", "all"):
        from pipeline.enhanced_features import stage_2b_enhanced
        stage_2b_enhanced()
    if args.stage in ("3", "all"):
        from pipeline.classification import stage_3_classify
        stage_3_classify()
    if args.stage in ("3b", "all"):
        from pipeline.experiments import stage_3b_experiments
        stage_3b_experiments()
    if args.stage in ("4", "all"):
        from pipeline.sequence_models import stage_4_sequence
        stage_4_sequence()
    if args.stage in ("6", "all"):
        from pipeline.oos_evaluation import stage_6_oos
        stage_6_oos()
    if args.stage == "5":
        from apps.chatbot import stage_5_chatbot
        stage_5_chatbot(mode=args.chat_mode)


if __name__ == "__main__":
    main()
