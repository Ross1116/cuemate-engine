package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"

	scoringv1 "github.com/Ross1116/cuemate-engine/go/gen/djengine/scoring/v1"
	"github.com/Ross1116/cuemate-engine/go/internal/scoringclient"
	"google.golang.org/protobuf/encoding/protojson"
)

func main() {
	os.Exit(run(os.Args[1:]))
}

func run(args []string) int {
	if len(args) == 0 {
		printUsage(os.Stderr)
		return 1
	}

	switch args[0] {
	case "metadata":
		return runMetadata(args[1:])
	case "score":
		return runScore(args[1:])
	default:
		printUsage(os.Stderr)
		return 1
	}
}

func runMetadata(args []string) int {
	fs := flag.NewFlagSet("metadata", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	addr := fs.String("addr", "", "Override scoring gRPC address.")
	jsonOutput := fs.Bool("json", false, "Emit JSON instead of text.")
	if err := fs.Parse(args); err != nil {
		return 1
	}

	client, err := newClient(*addr)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	defer client.Close()

	resp, err := client.GetScoringMetadata(context.Background(), &scoringv1.GetScoringMetadataRequest{})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}

	if *jsonOutput {
		marshaler := protojson.MarshalOptions{Indent: "  ", UseProtoNames: true}
		payload, err := marshaler.Marshal(resp)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		fmt.Println(string(payload))
		return 0
	}

	fmt.Printf("engine_version: %s\n", resp.GetEngineVersion())
	fmt.Printf("analysis_signature: %s\n", resp.GetActiveSignatures().GetAnalysisSignature())
	fmt.Printf("config_signature: %s\n", resp.GetActiveSignatures().GetConfigSignature())
	fmt.Printf("scoring_contract_id: %s\n", resp.GetActiveSignatures().GetScoringContractId())
	fmt.Printf("components: %d\n", len(resp.GetComponents()))
	return 0
}

func runScore(args []string) int {
	fs := flag.NewFlagSet("score", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	addr := fs.String("addr", "", "Override scoring gRPC address.")
	fixturePath := fs.String("fixture", "", "Path to a JSON-encoded ScoreCandidateRequest fixture.")
	jsonOutput := fs.Bool("json", false, "Emit JSON instead of text.")
	if err := fs.Parse(args); err != nil {
		return 1
	}
	if *fixturePath == "" {
		fmt.Fprintln(os.Stderr, "--fixture is required")
		return 1
	}

	request, err := loadScoreFixture(*fixturePath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}

	client, err := newClient(*addr)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	defer client.Close()

	resp, err := client.ScoreCandidate(context.Background(), request)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}

	if *jsonOutput {
		marshaler := protojson.MarshalOptions{Indent: "  ", UseProtoNames: true}
		payload, err := marshaler.Marshal(resp)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		fmt.Println(string(payload))
		return 0
	}

	scored := resp.GetScoredCandidate()
	fmt.Printf("candidate: %s\n", scored.GetCandidate().GetTrackId())
	fmt.Printf("score: %.4f\n", scored.GetFinalScore())
	fmt.Printf("move: %s\n", scored.GetMove())
	fmt.Printf("risk: %s\n", scored.GetRisk())
	return 0
}

func newClient(addrOverride string) (scoringclient.Client, error) {
	cfg, err := scoringclient.LoadConfig()
	if err != nil {
		return nil, err
	}
	if addrOverride != "" {
		cfg.Addr = addrOverride
	}
	return scoringclient.New(cfg)
}

func loadScoreFixture(path string) (*scoringv1.ScoreCandidateRequest, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read fixture: %w", err)
	}

	var request scoringv1.ScoreCandidateRequest
	if err := protojson.Unmarshal(raw, &request); err != nil {
		return nil, fmt.Errorf("decode score fixture: %w", err)
	}
	if request.GetCurrentTrack().GetTrackId() == "" || request.GetCandidate().GetTrackId() == "" {
		return nil, errors.New("fixture must provide current_track.track_id and candidate.track_id")
	}
	return &request, nil
}

func printUsage(w *os.File) {
	fmt.Fprintln(w, "usage:")
	fmt.Fprintln(w, "  scoringctl metadata [--addr host:port] [--json]")
	fmt.Fprintln(w, "  scoringctl score --fixture path [--addr host:port] [--json]")
}
