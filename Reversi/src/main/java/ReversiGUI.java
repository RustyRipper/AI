import javax.swing.*;
import java.awt.*;
import java.awt.event.*;

public class ReversiGUI {
    private final JFrame frame;
    private final Reversi reversi;
    private final JButton[][] buttons;

    public ReversiGUI() {
        frame = new JFrame("Reversi");
        reversi = new Reversi();
        buttons = new JButton[8][8];
        Timer computerMoveTimer;

        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 800);
        frame.setLayout(new GridLayout(8, 8));

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                buttons[i][j] = new JButton();
                buttons[i][j].setFont(new Font("Arial", Font.BOLD, 48));
                buttons[i][j].setEnabled(false);
                buttons[i][j].addActionListener(new ButtonClickListener(i, j));
                buttons[i][j].setBackground(new Color(0, 100, 0));
                frame.add(buttons[i][j]);
            }
        }
        updateBoard();

        frame.setVisible(true);
        computerMoveTimer = new Timer(100, e -> {
            ReversiMiniMax.MoveResult result;
            ReversiMiniMax reversiMiniMax = new ReversiMiniMax(reversi.currentPlayer);
            if (reversi.currentPlayer == 'B') {
                result = reversiMiniMax.minimaxalphabeta(
                        new Reversi(Main.cloneCharArray(reversi.board), reversi.currentPlayer),
                        reversiMiniMax.playerMove,
                        5,
                        Integer.MIN_VALUE,
                        Integer.MAX_VALUE);
            } else {
                result = reversiMiniMax.minimaxalphabeta(
                        new Reversi(Main.cloneCharArray(reversi.board), reversi.currentPlayer),
                        reversiMiniMax.playerMove,
                        4,
                        Integer.MIN_VALUE,
                        Integer.MAX_VALUE);
            }
            if (result.row >= 0 && result.col >= 0 && reversi.isValidMove(result.row, result.col, reversi.currentPlayer)) {
                System.out.println("Ruch gracza " + reversi.currentPlayer);
                //Thread.sleep(2000);
                reversi.makeMove(result.row, result.col, reversi.currentPlayer);
                updateBoard();
                reversi.displayBoard();
                buttons[result.row][result.col].setEnabled(false);
                buttons[result.row][result.col].setBackground(Color.BLUE);
                if (reversi.handleNoMoveAvailable(false))
                    showMessage("Result: B=" + reversi.countScore('B') + "   W=" + reversi.countScore('W'));

            } else if (result.row < 0 || result.col < 0) {
                reversi.switchPlayer();
            }
        });
        computerMoveTimer.start();
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
            if (reversi.noValidMove(reversi.currentPlayer))
                reversi.switchPlayer();
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

    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                new ReversiGUI();
            }
        });
    }
}
