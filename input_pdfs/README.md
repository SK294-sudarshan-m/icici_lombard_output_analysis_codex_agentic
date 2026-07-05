# Ground-truth PDF input

Place any number of reference PDFs in this folder. The analyzer scans the folder
recursively, so batches of 50-100 PDFs can be organized into subfolders without
changing the command. Ground-truth files are matched to output cases by AL number
when a text layer exists and otherwise by normalized source filename.

The staged `ground_truth_001.pdf` is the supplied 76-page reference document. Its
neutral filename avoids persisting a patient name in generated analysis artifacts.
