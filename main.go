package main

import (
	"bufio"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/richinsley/jumpboot"
)

//go:embed recall.py
var recallScript string

const envName = "jb-recall"
const pythonVersion = "3.11"

type RecallClient struct {
	process *jumpboot.PythonProcess
	reader  *bufio.Reader
	writer  io.Writer
}

type Message struct {
	Cmd        string   `json:"cmd,omitempty"`
	Status     string   `json:"status,omitempty"`
	Error      string   `json:"error,omitempty"`
	Reason     string   `json:"reason,omitempty"`
	Path       string   `json:"path,omitempty"`
	DbPath     string   `json:"db_path,omitempty"`
	Query      string   `json:"query,omitempty"`
	Limit      int      `json:"limit,omitempty"`
	Force      bool     `json:"force,omitempty"`
	Extensions []string `json:"extensions,omitempty"`
	Count      int      `json:"count,omitempty"`
	Indexed    int      `json:"indexed,omitempty"`
	Skipped    int      `json:"skipped,omitempty"`
	Chunks     int      `json:"chunks,omitempty"`
	Results    []Result `json:"results,omitempty"`
}

type Result struct {
	ID       string  `json:"id"`
	Score    float64 `json:"score"`
	Text     string  `json:"text"`
	Path     string  `json:"path"`
	Filename string  `json:"filename"`
	ChunkIdx int     `json:"chunk_idx"`
}

func NewRecallClient(rootDir string) (*RecallClient, error) {
	// Create or use existing environment
	env, err := jumpboot.CreateEnvironmentMamba(envName, rootDir, pythonVersion, "conda-forge", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create environment: %w", err)
	}

	// Install dependencies if new environment
	if env.IsNew {
		fmt.Fprintln(os.Stderr, "Installing dependencies (first run, may take a few minutes)...")
		err = env.PipInstallPackages([]string{"sentence-transformers", "chromadb", "torch"}, "", "", false, nil)
		if err != nil {
			return nil, fmt.Errorf("failed to install packages: %w", err)
		}
	}

	// Create program with embedded script
	cwd, _ := os.Getwd()
	program := &jumpboot.PythonProgram{
		Name: "jb-recall",
		Path: cwd,
		Program: jumpboot.Module{
			Name:   "__main__",
			Path:   filepath.Join(cwd, "recall.py"),
			Source: base64.StdEncoding.EncodeToString([]byte(recallScript)),
		},
	}

	// Start Python process
	process, _, err := env.NewPythonProcessFromProgram(program, nil, nil, false)
	if err != nil {
		return nil, fmt.Errorf("failed to start Python process: %w", err)
	}

	client := &RecallClient{
		process: process,
		reader:  bufio.NewReader(process.PipeIn),
		writer:  process.PipeOut,
	}

	// Forward stderr
	go io.Copy(os.Stderr, process.Stderr)

	// Wait for ready
	resp, err := client.recv()
	if err != nil {
		process.Terminate()
		return nil, fmt.Errorf("failed to get ready signal: %w", err)
	}
	if resp.Status != "ready" {
		process.Terminate()
		return nil, fmt.Errorf("unexpected status: %s", resp.Status)
	}

	return client, nil
}

func (c *RecallClient) send(msg Message) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	_, err = c.writer.Write(append(data, '\n'))
	return err
}

func (c *RecallClient) recv() (*Message, error) {
	line, err := c.reader.ReadString('\n')
	if err != nil {
		return nil, err
	}
	var msg Message
	err = json.Unmarshal([]byte(line), &msg)
	return &msg, err
}

func (c *RecallClient) Close() {
	c.send(Message{Cmd: "quit"})
	c.process.Terminate()
}

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	cmd := os.Args[1]

	// Root directory for jb-recall
	homeDir, _ := os.UserHomeDir()
	rootDir := filepath.Join(homeDir, ".jb-recall")

	// Create client
	client, err := NewRecallClient(rootDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	// Initialize database
	dbPath := filepath.Join(rootDir, "db")
	client.send(Message{Cmd: "init", DbPath: dbPath})
	initResp, err := client.recv()
	if err != nil || initResp.Status == "error" {
		fmt.Fprintf(os.Stderr, "Init error: %v %s\n", err, initResp.Error)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Database ready (%d chunks indexed)\n", initResp.Count)

	switch cmd {
	case "index":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: jb-recall index <path>")
			os.Exit(1)
		}
		path := os.Args[2]
		absPath, _ := filepath.Abs(path)
		info, err := os.Stat(absPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		force := contains(os.Args, "--force")

		if info.IsDir() {
			client.send(Message{Cmd: "index_dir", Path: absPath, Force: force})
		} else {
			client.send(Message{Cmd: "index_file", Path: absPath, Force: force})
		}

		resp, err := client.recv()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		if resp.Status == "error" {
			fmt.Fprintf(os.Stderr, "Error: %s\n", resp.Error)
			os.Exit(1)
		}

		if info.IsDir() {
			fmt.Printf("Indexed %d files (%d skipped)\n", resp.Indexed, resp.Skipped)
		} else {
			fmt.Printf("Status: %s\n", resp.Status)
			if resp.Chunks > 0 {
				fmt.Printf("Chunks: %d\n", resp.Chunks)
			}
		}

	case "search", "query", "q":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: jb-recall search <query>")
			os.Exit(1)
		}
		query := strings.Join(os.Args[2:], " ")

		client.send(Message{Cmd: "search", Query: query, Limit: 5})
		resp, err := client.recv()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		if resp.Status == "error" {
			fmt.Fprintf(os.Stderr, "Error: %s\n", resp.Error)
			os.Exit(1)
		}

		if len(resp.Results) == 0 {
			fmt.Println("No results found.")
		} else {
			for i, r := range resp.Results {
				fmt.Printf("\n--- Result %d (%.2f) ---\n", i+1, r.Score)
				fmt.Printf("File: %s\n", r.Filename)
				fmt.Printf("Path: %s\n", r.Path)
				text := r.Text
				if len(text) > 300 {
					text = text[:300] + "..."
				}
				fmt.Printf("Content:\n%s\n", text)
			}
		}

	case "stats":
		client.send(Message{Cmd: "stats"})
		resp, _ := client.recv()
		fmt.Printf("Indexed chunks: %d\n", resp.Count)

	case "clear":
		client.send(Message{Cmd: "clear"})
		client.recv()
		fmt.Println("Database cleared.")

	case "json":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: jb-recall json <query>")
			os.Exit(1)
		}
		query := strings.Join(os.Args[2:], " ")
		client.send(Message{Cmd: "search", Query: query, Limit: 10})
		resp, _ := client.recv()
		output, _ := json.MarshalIndent(resp, "", "  ")
		fmt.Println(string(output))

	default:
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println(`jb-recall - Semantic memory layer

Usage:
  jb-recall index <path>     Index a file or directory
  jb-recall search <query>   Search indexed content
  jb-recall stats            Show database statistics
  jb-recall clear            Clear the database
  jb-recall json <query>     Search and output JSON (for integration)

Examples:
  jb-recall index ~/clawd/memory
  jb-recall search "what did we discuss about FDA wrappers"
  jb-recall q moltbot migration`)
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}
