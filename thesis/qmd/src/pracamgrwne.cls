\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{pracamgr_wne}[2006/05/05 v0.5.0 Praca magisterska]


\def\@baseclass{report}
\def\@baseclass{article} % fix for qmd
\@ifundefined{l@chapter}{\newcounter{chapter}%
  \newcommand*\l@chapter{\@dottedtocline{0}{0em}{1.5em}}}{}
\def\@rodzajpracy{magisterska}
\DeclareOption{licencjacka}{\def\@rodzajpracy{licencjacka}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{\@baseclass}}
\PassOptionsToClass{a4paper,twoside,openright,12pt}{\@baseclass}
\ProcessOptions

\LoadClass{\@baseclass}

% List of Appendices, structured like \listoffigures/\listoftables below.
% Entries are registered per-appendix via \addcontentsline{lap}{section}{...}
% in 99-appendix.qmd. Chained onto \listoftables (rather than exposed as a
% separate Quarto option, since Pandoc's PDF template has no native "lap"
% front-matter switch) so it appears immediately after the existing
% List of Tables, on its own page.
%
% Extension is "lap" (not "loa"): the algorithm package (loaded in
% masters-thesis.qmd's header-includes for the pseudocode-block algorithms)
% already writes its own List-of-Algorithms entries to a .loa file via every
% \caption{...} inside an algorithm environment. Reusing "loa" here silently
% merged those algorithm captions into this list, run together with no line
% breaks or page numbers since they weren't logged at the "section" level
% tocloft formats.
\usepackage{tocloft}
\newlistof{appendices}{lap}{List of Appendices}

% tocloft's own \listoffigures/\listoftables definitions are deferred via
% \AtBeginDocument (see tocloft.sty), which runs after any ordinary preamble
% \renewcommand -- so redefining these two commands here must ALSO be
% deferred via \AtBeginDocument, registered after \usepackage{tocloft} above,
% so this hook fires after tocloft's own and therefore wins.
\AtBeginDocument{%
  \let\oldlistoffigures\listoffigures
  \renewcommand{\listoffigures}{%
    \cleardoublepage
    \oldlistoffigures
  }
  \let\oldlistoftables\listoftables
  \renewcommand{\listoftables}{%
    \oldlistoftables
    \cleardoublepage
    \listofappendices
  }
}
\usepackage{amsmath}
\DeclareMathOperator*{\argmax}{argmax}
\DeclareMathOperator*{\argmin}{argmin}
\textwidth\paperwidth
\advance\textwidth -50mm
\oddsidemargin-1in
\advance\oddsidemargin 25mm
\evensidemargin-1in
\advance\evensidemargin 25mm
\topmargin -1in
\advance\topmargin 25mm
% No header content is ever displayed, but article.cls's default
% \headheight/\headsep still consume vertical space above \textheight unless
% zeroed -- previously pushing the printed top margin to ~37mm instead of the
% required 25mm. \footskip is sized to hold the 12pt footer page number
% within the 25mm bottom margin rather than beyond it.
\headheight 0pt
\headsep 0pt
\footskip 12mm
\textheight\paperheight
\advance\textheight -50mm
\advance\textheight -\footskip
\marginparwidth15mm
% Define abstract environment with large font
% The \null\vfil ... \vfil\null pair previously wrapping this body is the
% standard "vertically centre the abstract on its own page" idiom. It only
% works when the abstract is short enough to sit on one page with the glue
% around it. Once the text grew past that, the leading \vfil stretched to
% fill the whole page, stranding the "Abstract" heading alone on one page
% and pushing the body onto the next, itself indented from the top by the
% trailing \vfil. WNE UW rule 1c expects the abstract, Erasmus codes, and
% thematic classification to share a single page, so the abstract starts at
% the top of the page and flows normally instead.
% Body text stays at the document's 12 pt (WNE UW rule 2). The \large here
% previously rendered the abstract at 14.3 pt against 12 pt body text. Every
% font-size deviation the requirements allow is downward -- 10 pt for
% footnotes (rule 8), figure sources (rule 11), and long block quotations
% (rule 7) -- and none licenses a larger font, so there is no basis for
% setting the abstract above 12 pt.
\renewenvironment{abstract}{%
  \begin{center}
    \bfseries \abstractname
  \end{center}
  \quotation
}{%
  \endquotation
}

\renewcommand\maketitle{%
  \begin{titlepage}%
    \let\footnotesize\small
    \let\footnoterule\relax
    \let \footnote \thanks
    \begin{center}%
      {\LARGE\textbf{University of Warsaw}\\
        Faculty of Economic Sciences\par}
      \vspace{1cm plus 1fill}
      {\Large\bf\@author\par}
      \vspace{0.2cm}
      {\large Nr albumu: \@nralbumu\par}
      \vspace{8mm plus .1fill}
      {\Huge\textbf{\@title}\par}
      \vspace{8mm plus .1fill}
      {\large\bf Praca \@rodzajpracy\\[3pt]
        na kierunku \MakeUppercase{\@speciality} \\
%----zakres---
        \@zakres \par}
      \vspace{2cm plus 1.5fill}
      \begin{flushright}\large
        \begin{tabular}{l}
          The thesis written under the supervision of\\
          \bfseries \@opiekun
        \end{tabular}
      \end{flushright}
      \vspace{1cm plus 1fill}
      {\large \@date\par}
    \end{center}
    \@thanks
  \end{titlepage}%
  \begin{titlepage}
    \c@page=1
    \thispagestyle{plain}
    \large
    \null
    \vfill


   \textbf{\Large }
    \vspace{10mm}
    \vspace{10mm}

   Data \hfill Podpis opiekuna

   \vspace{2cm}
   \textbf{\Large Statement of the Supervisor on Submission of the Thesis}
   \vspace{10mm}

    I hereby certify that the thesis submitted has been prepared under my supervision and I declare that it satisfies the requirements of submission in the
    proceedings for the award of a degree.

   Date \hfill Signature of the Supervisor:
   \vspace{10mm}

   \vspace{2cm}
   \textbf{\Large O\'swiadczenie autora (autor\'ow) pracy}
    \vspace{10mm}

   \vspace{10mm}


   Data \hfill Podpis autora (autor\'ow) pracy

   \vspace{2cm}
   \textbf{\Large Statement of the Author(s) on Submission of the Thesis }
      \vspace{10mm}

   Aware of legal liability I certify that the thesis submitted has been prepared by myself and does not include information gathered contrary to the law.

   I also declare that the thesis submitted has not been the subject of proceedings resulting in the award of a university degree.

  Furthermore I certify that the submitted version of the thesis is identical with its attached electronic version.


Date \hfill Signature of the Author(s) of the thesis

   \vspace{10mm}


    \vspace{4cm}
  \end{titlepage}
  \setcounter{footnote}{0}%
  \setcounter{chapter}{1}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}
\def\nralbumu#1{\gdef\@nralbumu{#1}}
\def\@nralbumu{???\ClassError{pracamgr}{Brak numeru albumu}\@ehc}
\def\speciality#1{\gdef\@speciality{#1}}
\def\@speciality{???\ClassError{pracamgr}{Nie podano kierunku studiow}\@ehc}
%----zakres niekonieczny-----
\def\zakres#1{\gdef\@zakres{w zakresie \MakeUppercase{#1}}}
\def\@zakres{}
\def\opiekun#1{\gdef\@opiekun{#1}}
\def\@opiekun{???\ClassError{pracamgr}{Brak danych opiekuna pracy}\@ehc}
\def\keywords#1{\gdef\@keywords{#1}}
\def\@keywords{???\ClassError{pracamgr}{Brak slow kluczowych}\@ehc}
\def\dziedzina#1{\gdef\@dziedzina{#1}}
\def\@dziedzina{???\ClassError{pracamgr}{Brak dziedziny Socrates-Erasmus}\@ehc}
\def\klasyfikacja#1{\gdef\@klasyfikacja{#1}}
\def\@klasyfikacja{???\ClassError{pracamgr}{Brak klasyfikacji tematycznej}\@ehc}
%-------------------nowe------------
\def\tytulang#1{\gdef\@tytulang{#1}}
\def\@tytulang{???\ClassError{pracamgr}{Brak tytulu po angielsku}\@ehc}


\newenvironment{summary}{%
  \titlepage
  \null\nobreak\vfil
  \@beginparpenalty\@lowpenalty
  \begin{center}%
    \bfseries\large Summary
    \@endparpenalty\@M
  \end{center}}%
{\par
  \vspace*{26pt}%
  \begin{center}%
    \bfseries\large Key words*
    \@endparpenalty\@M
  \end{center}
  \@keywords\par
  \vspace*{26pt}%
  \begin{center}%
    \bfseries\large Area of study (codes according to Erasmus Subject Area Codes List)
    \@endparpenalty\@M
  \end{center}
   \@dziedzina\par
  \vspace*{26pt}%
  \begin{center}%
    \bfseries\large Theme classification
    \@endparpenalty\@M
  \end{center}
  \@klasyfikacja\par
  \vspace*{26pt}%
%-------------------nowosc----------------
  \begin{center}%
    \bfseries\large The title of the thesis in Polish
    \@endparpenalty\@M
  \end{center}
  \@tytulang\par
  \nobreak\vfil\null\endtitlepage\cleardoublepage}

\def\cleardoublepage{\clearpage\if@twoside \ifodd\c@page\else
    \hbox{}\thispagestyle{empty}\newpage\if@twocolumn\hbox{}\newpage\fi\fi\fi}

\renewcommand*\@seccntformat[1]{\csname the#1\endcsname.\enspace}
\def\numberline#1{\hb@xt@\@tempdima{#1.\hfil}}
\renewcommand*\l@chapter[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 1.0em \@plus\p@
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\mdseries
      \leaders\hbox{$\m@th
        \mkern \@dotsep mu\hbox{.}\mkern \@dotsep
        mu$}\hfill
      \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}

\endinput
