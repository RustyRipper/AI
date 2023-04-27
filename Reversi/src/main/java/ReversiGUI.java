

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.ArrayList;
import java.util.Random;

public class ReversiGUI {
    private final JFrame frame;
    private final Reversi reversi;
    private final JButton[][] buttons;
    static ArrayList<Integer> listNodesVisited = new ArrayList();
    static ArrayList<Long> listTime = new ArrayList();

    static int BWin = 0;
    static int WWin = 0;

    public ReversiGUI() throws InterruptedException {
        frame = new JFrame("Reversi");

        reversi = new Reversi();
        prepareReversiCustomBoard(reversi);
        buttons = new JButton[8][8];
        Timer computerMoveTimer;

        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 800);
        frame.setLayout(new GridLayout(8, 8));

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                buttons[i][j] = new JButton();
                buttons[i][j].setEnabled(false);
                buttons[i][j].addActionListener(new ButtonClickListener(i, j));
                buttons[i][j].setBackground(new Color(0, 100, 0));
                frame.add(buttons[i][j]);
            }
        }
        updateBoard();



        frame.setVisible(true);
        computerMoveTimer = new Timer(1, e -> {
            ReversiMiniMax.MoveResult result = null;
            ReversiMiniMax reversiMiniMax = new ReversiMiniMax(reversi.currentPlayer);
            if (reversi.currentPlayer == 'B') {
                result = reversiMiniMax.minimaxalphabeta(
                        new Reversi(Main.cloneCharArray(reversi.board), reversi.currentPlayer),
                        reversiMiniMax.playerMove,
                        8,
                        Integer.MIN_VALUE,
                        Integer.MAX_VALUE,
                        "points");
            } else {
                result = reversiMiniMax.minimaxalphabeta(
                        new Reversi(Main.cloneCharArray(reversi.board), reversi.currentPlayer),
                        reversiMiniMax.playerMove,
                        8,
                        Integer.MIN_VALUE,
                        Integer.MAX_VALUE,
                        "dynamic");

            }

            if (result != null && result.row >= 0 && result.col >= 0 && reversi.isValidMove(result.row, result.col, reversi.currentPlayer)) {

                listNodesVisited.add(result.nodeVisited);
                listTime.add(result.time);
                //System.out.println("Odwiedzone węzły: " + result.nodeVisited);
                //System.out.println("Czas: " + result.time);
                //System.out.println("Ruch gracza " + reversi.currentPlayer);
                //Thread.sleep(2000);
                reversi.makeMove(result.row, result.col, reversi.currentPlayer);
                updateBoard();
                //reversi.displayBoard();
                buttons[result.row][result.col].setEnabled(false);
                buttons[result.row][result.col].setBackground(Color.BLUE);
                if (reversi.gameOver){
                    //showMessage("Result: B=" + reversi.countScore('B') + "   W=" + reversi.countScore('W'));
                    if(reversi.countScore('B')> reversi.countScore('W')){
                        BWin++;
                    }
                    else WWin++;
                    //showMessage("AVG nodesVisited " + avg(listNodesVisited));
                    //showMessage("AVG Time " + avgL(listTime));
                    SwingUtilities.invokeLater(new Runnable() {
                        public void run() {
                            try {
                                new ReversiGUI();
                            } catch (InterruptedException e) {
                                throw new RuntimeException(e);
                            }
                        }
                    });
                    System.out.println("AVG Time " + avgL(listTime));
                    System.out.println("AVG nodes " + avg(listNodesVisited));
                    System.out.println("BWin " + BWin);
                    System.out.println("WWin " + WWin);
                }


            }
//            try {
//                //Thread.sleep(20000);
//            } catch (InterruptedException ex) {
//                throw new RuntimeException(ex);
//            }
        });

        computerMoveTimer.start();
    }

    private void prepareReversiCustomBoard(Reversi reversi) {
        Random rand = new Random();
        for (int i = 0; i < rand.nextInt(40) + 1; i++) {
            ArrayList<Move> moves = ReversiMiniMax.generateMoves(reversi.board, reversi.currentPlayer);
            Move move = moves.get(rand.nextInt(moves.size()));
                reversi.makeMove(move.row, move.col, reversi.currentPlayer);
        }
    }

    private class ButtonClickListener implements ActionListener {
        private final int row;
        private final int col;

        public ButtonClickListener(int row, int col) {
            this.row = row;
            this.col = col;
        }

        public void actionPerformed(ActionEvent event) {
            System.out.println("Ruch gracza " + reversi.currentPlayer);
            if (reversi.isValidMove(row, col, reversi.currentPlayer)) {
                reversi.makeMove(row, col, reversi.currentPlayer);
                updateBoard();
                if (reversi.handleNoMoveAvailable(false))
                    showMessage("Result: B=" + reversi.countScore('B') + "   W=" + reversi.countScore('W'));
            }
        }
    }

    private void updateBoard() {

        char[][] board = reversi.board;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if (board[i][j] == 'B') {
                    buttons[i][j].setEnabled(false);
                    buttons[i][j].setBackground(Color.BLACK);
                } else if (board[i][j] == 'W') {
                    buttons[i][j].setEnabled(false);
                    buttons[i][j].setBackground(Color.WHITE);
                } else {
                    buttons[i][j].setText("");
                    if (reversi.isValidMove(i, j, reversi.currentPlayer)) {
                        buttons[i][j].setEnabled(true);
                        buttons[i][j].setBackground(Color.PINK);
                    } else {
                        buttons[i][j].setEnabled(false);
                        buttons[i][j].setBackground(Color.GRAY);
                    }
                }
            }
        }
        String currentPlayer = reversi.currentPlayer == 'B' ? "Black" : "White";
        frame.setTitle("Reversi - Current Player: " + currentPlayer);
    }

    private void showMessage(String message) {
        JOptionPane.showMessageDialog(frame, message, "Info", JOptionPane.INFORMATION_MESSAGE);
    }

    public static double avg(ArrayList<Integer> list) {
        int suma = 0;
        for (int i = 0; i < list.size(); i++) {
            suma += list.get(i);
        }
        double avg = (double) suma / list.size();
        return avg;
    }
    public static double avgL(ArrayList<Long> list) {
        int suma = 0;
        for (int i = 0; i < list.size(); i++) {
            suma += list.get(i);
        }
        double avg = (double) suma / list.size();
        return avg;
    }


    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                try {
                    new ReversiGUI();
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        });
    }
}
